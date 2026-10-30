// SPDX-License-Identifier: GPL-3.0-or-later

//! Element provenance for mesh-rebuilding modifiers.
//!
//! A modifier that rebuilds topology reports where each output vertex and polygon came from.
//! Edges and polygon corners are derived from those origins, so [`remap_attributes`] is the
//! single place that rewrites ID-keyed attributes after a rebuild. Vertex origins are weighted
//! so interpolated data (vertex groups, POINT attributes) blends like Blender's `CustomData`.

use std::collections::{HashMap, HashSet};
use std::fmt::Display;
use std::hash::Hash;
use std::str::FromStr;

use glam::{DVec3, Vec3};
use serde_json::{Map, Value, json};

use crate::error::{ErrorCode, PotError, Result};

use super::attribute_keys;
use super::clnor::{self, NormalCodeOptions};
use super::color_layers::{self, ColorDomain, ColorLayer};
use super::interpolation::interp_weights_poly_v2;
use super::uv_layers::{self, UvLayer};
use super::{Mesh, ORIGINDEX_ATTRIBUTE, ORIGINDEX_NONE, edge_key};

/// One ordered `BMesh` point-data interpolation callback during Bevel.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct PointInterpolationEvent {
    /// Output vertex written by the callback.
    pub output_vertex: usize,
    /// Input vertex also updated when the callback target is an original `BMesh` vertex.
    pub mutated_source_vertex: Option<usize>,
    /// Input point values and weights read from the source face's loops.
    pub sources: Vec<(usize, f64)>,
}

/// Identity of one source FLOAT2 corner layer that Blender merges independently.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum CornerFloat2LayerKey {
    /// Canonical `uv_layers` store layer.
    Uv(String),
    /// Top-level native or named mesh attribute storage key.
    Attribute(String),
    /// Raw `blender_attributes` entry name.
    BlenderAttribute(String),
}

/// Per-output-corner source-loop origins after merging one FLOAT2 layer.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct CornerFloat2Sources {
    /// Which source FLOAT2 layer has these corner origins.
    pub key: CornerFloat2LayerKey,
    /// One source-loop weight list per output corner, in output loop order.
    pub corners: Vec<Vec<(usize, f64)>>,
}

/// Where each element of a rebuilt mesh came from, indexed by output storage order.
#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct Origin {
    /// Per output vertex: input vertex storage indices with weights summing to one.
    /// An empty list marks a new vertex without input data.
    pub vertices: Vec<Vec<(usize, f64)>>,
    /// Per output vertex: the input storage index whose `blender_origindex` value it takes, or
    /// `None` for Blender `ORIGINDEX_NONE`. When `Some`, it overrides the weight-one rule of
    /// [`carrier_source`]; weighted attribute provenance is unaffected.
    pub source_vertices: Option<Vec<Option<usize>>>,
    /// Per output polygon, in the order passed to `build_mesh`: the input polygon storage index,
    /// or `None` for a generated polygon.
    pub faces: Vec<Option<usize>>,
    /// Explicit source corners per output corner, in output order, as `(input corner index, weight)`
    /// with the input corner index counted as in [`corner_offsets`]. `None` derives corner sources
    /// from the vertex origins restricted to each output face's source polygon.
    pub corners: Option<Vec<Vec<(usize, f64)>>>,
    /// Ordered `BM_loop_interp_from_face` writes for point-data attributes. `Some(empty)` means
    /// Bevel performs no callback for any output point; `None` means no explicit event stream.
    pub point_interpolation_events: Option<Vec<PointInterpolationEvent>>,
    /// Per output loop, whether Bevel invoked source-face interpolation for corner data. `Some`
    /// only when source CORNER quaternion attributes need copy distinction.
    pub corner_interpolated: Option<Vec<bool>>,
    /// Merged source corners per FLOAT2 layer; Blender computes connectivity independently per layer.
    pub corner_float2_sources: Option<Vec<CornerFloat2Sources>>,
}

impl Origin {
    /// Origin where each output vertex copies one input vertex, or is new when `None`.
    /// `source_vertices` is `None`; [`carrier_source`] then uses the weight-one rule.
    pub(crate) fn copies(
        vertices: impl IntoIterator<Item = Option<usize>>,
        faces: Vec<Option<usize>>,
    ) -> Self {
        let vertices = vertices
            .into_iter()
            .map(|source| source.map(|index| vec![(index, 1.0)]).unwrap_or_default())
            .collect();
        Self {
            vertices,
            source_vertices: None,
            faces,
            corners: None,
            ..Default::default()
        }
    }
}

/// The input storage index whose `blender_origindex` value output vertex `vertex` takes, or `None`
/// for `ORIGINDEX_NONE`. An exact source wins; else a single weight-one source is used.
fn carrier_source(origin: &Origin, vertex: usize) -> Option<usize> {
    if let Some(exact) = &origin.source_vertices {
        return exact[vertex];
    }
    match origin.vertices[vertex].as_slice() {
        [(index, weight)] if weight.to_bits() == 1.0_f64.to_bits() => Some(*index),
        _ => None,
    }
}

/// One Blender mixing stage of a face-sampled element: `sources` mixed into one value, then weighted by
/// `weight` in the element's outer mix. Blender converts each stage to the attribute type before the outer
/// mix (`vert_interpolation_from_corner`, `subdiv_mesh.cc:498`; `subdiv_vert_data_interpolate`, `subdiv_mesh.cc:971`).
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Stage {
    pub weight: f64,
    pub sources: Vec<(usize, f64)>,
}

/// How a subdivided vertex mixes its sources in Blender. A face-sampled vertex mixes its slots: each slot is
/// converted to the attribute type first, then the slots are mixed by their outer weights (`mix4`). A loose-edge
/// vertex is one two-source `mix2` with no outer mix (`subdiv_mesh_vert_of_loose_edge_interpolate`, `subdiv_mesh.cc:1434`).
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum Mixing {
    /// Slots as [`Stage`]s, mixed by their outer weights.
    Slots(Vec<Stage>),
    /// Two `(source, weight)` sources; the second weight is the factor of `mix2`.
    Pair(Vec<(usize, f64)>),
}

/// Staged mixes of a subdivided mesh, indexed like the flat [`Origin`] lists. A vertex without a mixing
/// (`None`) is a coarse copy, whose flat source is the copied vertex.
#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct Staged {
    pub vertices: Vec<Option<Mixing>>,
    /// `None` when the output carries no corner data, as for [`Origin::corners`].
    pub corners: Option<Vec<Vec<Stage>>>,
}

impl Staged {
    /// Flat sources of the stages, one `(source, weight)` per source, merged by source.
    pub(crate) fn flatten(stages: &[Stage]) -> Vec<(usize, f64)> {
        let mut pairs: Vec<(usize, f64)> = stages
            .iter()
            .flat_map(|stage| {
                stage
                    .sources
                    .iter()
                    .map(move |&(source, weight)| (source, weight * stage.weight))
            })
            .collect();
        pairs.sort_by_key(|&(source, _)| source);
        let mut merged: Vec<(usize, f64)> = Vec::with_capacity(pairs.len());
        for (source, weight) in pairs {
            match merged.last_mut() {
                Some((last, total)) if *last == source => *total += weight,
                _ => merged.push((source, weight)),
            }
        }
        merged.retain(|(_, weight)| *weight > 0.0);
        merged
    }
}

/// How one output element mixes its sources, as Blender's attribute math does: flat weighted sources
/// (`mix_indices`), Blender's staged slots, or one two-source `mix2`.
#[derive(Clone, Copy)]
pub(crate) enum Mix<'a> {
    Flat(&'a [(usize, f64)]),
    Staged(&'a [Stage]),
    Pair(&'a [(usize, f64)]),
}

/// The mix of an element with `mixing` (its staged or pair mixing, if any), otherwise its flat `sources`.
pub(crate) fn element_mix<'a>(mixing: Option<&'a Mixing>, sources: &'a [(usize, f64)]) -> Mix<'a> {
    match mixing {
        Some(Mixing::Slots(stages)) => Mix::Staged(stages),
        Some(Mixing::Pair(pair)) => Mix::Pair(pair),
        None => Mix::Flat(sources),
    }
}

/// Rewrite the ID-keyed attributes of `output` (rebuilt from `source`) through `origin`.
///
/// `vertex_groups`, `crease_edge`, skin vertex keys, every `uv_layers` layer, every canonical color
/// layer, and every element-domain `mesh.attributes` entry are remapped. Corner data blends over the
/// source corners of each output corner (see [`corner_sources`]), so UVs follow the face corner they
/// came from. Shading flags follow the same origins: `shade_smooth` and `blender_polygon_smooth` copy
/// their source face's flag, and `sharp` and `seam` OR over source edges. Canonical `ColorStore` layers
/// preserve order/designations and use their Blender `FLOAT_COLOR/BYTE_COLOR` mixer. `custom_normal`
/// codes are never interpolated; [`crate::geom::clnor::transfer_custom_normals`] handles them.
/// Elements without a source get no entry, so readers use their type's default; no key points to a stale element.
///
/// # Errors
///
/// Returns `INTERNAL_ERROR` when `origin` does not describe `output`'s vertices and polygons.
pub(crate) fn remap_attributes(source: &Mesh, output: &mut Mesh, origin: &Origin) -> Result<()> {
    remap_attributes_skipping(source, output, origin, &[])
}

/// [`remap_attributes`] without the attributes named in `skipped`, which `output` does not receive.
/// A caller that rebuilds one of them from `source` skips it here, so no work runs on it and it does
/// not make corners need sources.
///
/// # Errors
///
/// Returns `INTERNAL_ERROR` when `origin` does not describe `output`'s vertices and polygons.
pub(crate) fn remap_attributes_skipping(
    source: &Mesh,
    output: &mut Mesh,
    origin: &Origin,
    skipped: &[&str],
) -> Result<()> {
    remap_attributes_internal(
        source,
        output,
        origin,
        None,
        None,
        skipped,
        None,
        false,
        None,
        [0.0, 0.0, 0.0, 1.0],
    )
}

/// Interpolated corner booleans use Blender's weighted `DefaultMixer<bool>`; ordinary topology group
/// merges continue to OR only positive-weight sources.
/// Boolean points without an origin use zero-initialized color attributes, as Blender's exact gather does.
pub(crate) fn remap_interpolated_corners_skipping(
    source: &Mesh,
    output: &mut Mesh,
    origin: &Origin,
    skipped: &[&str],
) -> Result<()> {
    remap_attributes_internal(
        source,
        output,
        origin,
        None,
        None,
        skipped,
        None,
        true,
        None,
        [0.0, 0.0, 0.0, 0.0],
    )
}

/// [`remap_attributes_skipping`] for a subdivided mesh whose face-sampled elements mix in Blender's stages.
/// Integer, quaternion, and byte data mix stage by stage through `staged`; other data uses `origin`. Each
/// output edge's source edges are given explicitly in `edges`, in `output.edges` order.
///
/// # Errors
///
/// Returns `INTERNAL_ERROR` when `origin`, `staged`, or `edges` does not describe `output`.
pub(crate) fn remap_attributes_staged(
    source: &Mesh,
    output: &mut Mesh,
    origin: &Origin,
    staged: Option<&Staged>,
    edges: Vec<Vec<(usize, f64)>>,
    skipped: &[&str],
) -> Result<()> {
    remap_attributes_internal(
        source,
        output,
        origin,
        staged,
        Some(edges),
        skipped,
        None,
        false,
        None,
        [0.0, 0.0, 0.0, 1.0],
    )
}

/// [`remap_attributes_skipping`] whose custom-normal loop basis is `corner_order` (a Blender loop
/// ordinal per output corner) when given. A modifier that renumbers loops itself, such as Array's
/// copies, passes its numbering here; `None` keeps the basis the source implies.
///
/// # Errors
///
/// Returns `INTERNAL_ERROR` when `origin` does not describe `output`'s vertices and polygons.
pub(crate) fn remap_attributes_with_basis(
    source: &Mesh,
    output: &mut Mesh,
    origin: &Origin,
    skipped: &[&str],
    corner_order: Option<&[usize]>,
) -> Result<()> {
    remap_attributes_internal(
        source,
        output,
        origin,
        None,
        None,
        skipped,
        corner_order,
        false,
        None,
        [0.0, 0.0, 0.0, 1.0],
    )
}

/// [`remap_attributes`] with each output edge's source edges given explicitly, in `output.edges`
/// order. An empty entry is an edge with no source (a cut span or a chord) and takes the default for
/// every edge attribute; [`remap_attributes`] derives the sources from the vertex origins instead.
///
/// # Errors
///
/// Returns `INTERNAL_ERROR` when `origin` or `edges` does not describe `output`.
pub(crate) fn remap_attributes_with_edges(
    source: &Mesh,
    output: &mut Mesh,
    origin: &Origin,
    edges: Vec<Vec<(usize, f64)>>,
) -> Result<()> {
    remap_attributes_internal(
        source,
        output,
        origin,
        None,
        Some(edges),
        &[],
        None,
        false,
        None,
        [0.0, 0.0, 0.0, 1.0],
    )
}
pub(crate) fn copy_direct_matrix_edge_attributes(
    source: &Mesh,
    output: &mut Mesh,
    origin: &Origin,
) -> Result<()> {
    let Some(source_layers) = source
        .attributes
        .get("blender_attributes")
        .and_then(Value::as_array)
    else {
        return Ok(());
    };
    let Some(output_layers) = output
        .attributes
        .get_mut("blender_attributes")
        .and_then(Value::as_array_mut)
    else {
        return Ok(());
    };
    let source_vertex_indices = source
        .vertices
        .iter()
        .enumerate()
        .map(|(index, vertex)| (vertex.id, index))
        .collect::<HashMap<_, _>>();
    let source_edges = source
        .edges
        .iter()
        .enumerate()
        .filter_map(|(index, edge)| {
            Some((
                edge_key(
                    *source_vertex_indices.get(&edge.vertices[0])?,
                    *source_vertex_indices.get(&edge.vertices[1])?,
                ),
                index,
            ))
        })
        .collect::<HashMap<_, _>>();
    let output_vertex_indices = output
        .vertices
        .iter()
        .enumerate()
        .map(|(index, vertex)| (vertex.id, index))
        .collect::<HashMap<_, _>>();
    let output_edge_sources = output
        .edges
        .iter()
        .map(|edge| {
            let source_vertex = |id: u32| {
                output_vertex_indices
                    .get(&id)
                    .and_then(|index| origin.vertices.get(*index))
                    .filter(|sources| {
                        sources.len() == 1 && sources[0].1.to_bits() == 1.0_f64.to_bits()
                    })
                    .map(|sources| sources[0].0)
            };
            source_vertex(edge.vertices[0])
                .zip(source_vertex(edge.vertices[1]))
                .and_then(|(first, second)| source_edges.get(&edge_key(first, second)).copied())
        })
        .collect::<Vec<_>>();
    for source_layer in source_layers {
        if source_layer.get("domain").and_then(Value::as_str) != Some("EDGE")
            || !matches!(
                source_layer.get("data_type").and_then(Value::as_str),
                Some("FLOAT4X4" | "MATRIX")
            )
        {
            continue;
        }
        let name = source_layer
            .get("name")
            .and_then(Value::as_str)
            .ok_or_else(|| {
                PotError::new(
                    ErrorCode::InternalError,
                    "matrix edge attribute has no name",
                )
            })?;
        let source_values = source_layer
            .get("values")
            .and_then(Value::as_array)
            .ok_or_else(|| {
                PotError::new(
                    ErrorCode::InternalError,
                    "matrix edge attribute has no values",
                )
            })?;
        let Some(output_layer) = output_layers
            .iter_mut()
            .find(|layer| layer.get("name").and_then(Value::as_str) == Some(name))
        else {
            continue;
        };
        let output_values = output_layer
            .get_mut("values")
            .and_then(Value::as_array_mut)
            .ok_or_else(|| {
                PotError::new(
                    ErrorCode::InternalError,
                    "matrix edge attribute output values are malformed",
                )
            })?;
        for (output_index, source_index) in output_edge_sources.iter().enumerate() {
            if let Some(source_index) = source_index {
                let source_value = source_values.get(*source_index).ok_or_else(|| {
                    PotError::new(
                        ErrorCode::InternalError,
                        "matrix edge attribute source is out of range",
                    )
                })?;
                let output_value = output_values.get_mut(output_index).ok_or_else(|| {
                    PotError::new(
                        ErrorCode::InternalError,
                        "matrix edge attribute output is out of range",
                    )
                })?;
                *output_value = source_value.clone();
            }
        }
    }
    Ok(())
}

/// [`remap_attributes`] for a rebuilt mesh whose output points each descend from one source vertex:
/// `edge_identity` names that source vertex per output point, one entry per point. Point-domain values
/// normally use `origin`, but Bevel's quaternion events replay from these identities; edge-domain values
/// use the source edge between each output edge's endpoint identities.
///
/// # Errors
///
/// Returns `INTERNAL_ERROR` when `origin` does not describe `output`, or when its point events or corner
/// metadata do not match `source` and `output`.
pub(crate) fn remap_attributes_with_edge_identity(
    source: &Mesh,
    output: &mut Mesh,
    origin: &Origin,
    edge_identity: Option<&[usize]>,
) -> Result<()> {
    remap_attributes_internal(
        source,
        output,
        origin,
        None,
        None,
        &[],
        None,
        false,
        edge_identity,
        [0.0, 0.0, 0.0, 1.0],
    )
}

fn remap_attributes_internal(
    source: &Mesh,
    output: &mut Mesh,
    origin: &Origin,
    staged: Option<&Staged>,
    edges: Option<Vec<Vec<(usize, f64)>>>,
    skipped: &[&str],
    corner_order: Option<&[usize]>,
    weighted_corner_booleans: bool,
    edge_identity: Option<&[usize]>,
    color_default: [f64; 4],
) -> Result<()> {
    let corner_count: usize = output.faces.iter().map(|face| face.vertices.len()).sum();
    if staged.is_some_and(|staged| {
        staged.vertices.len() != output.vertices.len()
            || staged
                .corners
                .as_ref()
                .is_some_and(|corners| corners.len() != corner_count)
    }) {
        return Err(PotError::new(
            ErrorCode::InternalError,
            "staged modifier provenance does not describe the rebuilt mesh",
        ));
    }
    let source_vertices = source.vertices.len();
    let invalid_point_events = origin
        .point_interpolation_events
        .as_ref()
        .is_some_and(|events| {
            edge_identity
                .is_none_or(|identity| identity.iter().any(|vertex| *vertex >= source_vertices))
                || events.iter().any(|event| {
                    event.output_vertex >= output.vertices.len()
                        || event
                            .mutated_source_vertex
                            .is_some_and(|vertex| vertex >= source_vertices)
                        || event
                            .sources
                            .iter()
                            .any(|(vertex, _)| *vertex >= source_vertices)
                })
        });
    let invalid_corner_metadata = origin
        .corner_interpolated
        .as_ref()
        .is_some_and(|flags| flags.len() != corner_count)
        || origin.corner_float2_sources.as_ref().is_some_and(|layers| {
            layers
                .iter()
                .any(|layer| layer.corners.len() != corner_count)
        });
    if edge_identity.is_some_and(|identity| identity.len() != output.vertices.len())
        || origin.vertices.len() != output.vertices.len()
        || origin.faces.len() != output.faces.len()
        || origin.source_vertices.as_ref().is_some_and(|vertices| {
            vertices.len() != output.vertices.len()
                || vertices
                    .iter()
                    .flatten()
                    .any(|index| *index >= source.vertices.len())
        })
        || invalid_point_events
        || invalid_corner_metadata
    {
        return Err(PotError::new(
            ErrorCode::InternalError,
            "modifier provenance does not describe the rebuilt mesh",
        ));
    }
    if source.attributes.is_empty() {
        output.attributes.clear();
        return Ok(());
    }
    let kept = |key: &str| !skipped.contains(&key);
    let has = |key: &str| kept(key) && source.attributes.contains_key(key);
    let entries = if kept("blender_attributes") {
        source
            .attributes
            .get("blender_attributes")
            .and_then(Value::as_array)
            .map_or(&[][..], Vec::as_slice)
    } else {
        &[][..]
    };
    let needs_edges = has("crease_edge")
        || has("blender_edge_flags")
        || has("sharp")
        || entries.iter().any(|entry| text(entry, "domain") == "EDGE")
        || source
            .attributes
            .iter()
            .filter(|(key, _)| kept(key.as_str()))
            .any(|(_, value)| named_domain(value).as_deref() == Some("edge"));
    let needs_corners = needs_corner_sources(source, skipped);
    let output_vertex_index: HashMap<u32, usize> = if needs_edges || needs_corners {
        output
            .vertices
            .iter()
            .enumerate()
            .map(|(index, vertex)| (vertex.id, index))
            .collect()
    } else {
        HashMap::new()
    };
    let identity_vertices: Option<Vec<Vec<(usize, f64)>>> =
        edge_identity.map(|identity| identity.iter().map(|source| vec![(*source, 1.0)]).collect());
    let edge_vertices = identity_vertices.as_deref().unwrap_or(&origin.vertices);
    let edge_origin = match (needs_edges, edges) {
        (false, _) => Vec::new(),
        (true, Some(explicit)) if explicit.len() == output.edges.len() => explicit,
        (true, Some(_)) => {
            return Err(PotError::new(
                ErrorCode::InternalError,
                "modifier provenance does not describe the rebuilt edges",
            ));
        }
        (true, None) => edge_sources(source, output, edge_vertices, &output_vertex_index),
    };
    let derived_corners;
    let corner_origin: &[Vec<(usize, f64)>] = match &origin.corners {
        Some(corners) => corners,
        None if needs_corners => {
            derived_corners = corner_sources(source, origin, output, &output_vertex_index);
            &derived_corners
        }
        None => &[],
    };
    let sources = Sources {
        source,
        output,
        origin,
        staged,
        point_identity: edge_identity,
        edge_origin: &edge_origin,
        corner_origin,
        weighted_corner_booleans,
        color_default,
    };
    let color_store = has(color_layers::ATTRIBUTE).then(|| {
        let mut store = color_layers::read(source);
        store
            .layers
            .retain(|layer| kept(attribute_keys::storage_key(&layer.name).as_ref()));
        store.layers = store
            .layers
            .iter()
            .map(|layer| sources.color_layer(layer))
            .collect();
        store
    });
    let color_layer_keys: HashSet<String> = color_store
        .as_ref()
        .into_iter()
        .flat_map(|store| store.layers.iter())
        .map(|layer| attribute_keys::storage_key(&layer.name).into_owned())
        .collect();
    // Rebuild code state from source corner identity, including a basis-only layer.
    let has_custom_normal_state = kept("custom_normal")
        && (clnor::corner_codes(source)?.is_some() || clnor::corner_order(source)?.is_some());
    let code_sources: Option<Vec<Option<usize>>> = has_custom_normal_state.then(|| {
        corner_origin
            .iter()
            .map(|sources| heaviest_source(sources))
            .collect()
    });
    // The store's layers are rebuilt per corner; their names and active/render choices carry over.
    let uv_store = has(uv_layers::ATTRIBUTE).then(|| {
        let mut store = uv_layers::read(source);
        store.layers = store
            .layers
            .iter()
            .map(|layer| sources.uv_layer(layer))
            .collect();
        store
    });
    let mut attributes = Map::new();
    for (key, value) in &source.attributes {
        if !kept(key.as_str()) || color_layer_keys.contains(key) {
            continue;
        }
        if key == uv_layers::ATTRIBUTE {
            // Rebuilt after the loop from `uv_store`.
            continue;
        }
        if key == "custom_normal" {
            // The code state is rebuilt below from source corner identity, never interpolated.
            continue;
        }
        let remapped = match key.as_str() {
            "vertex_groups" => value
                .as_object()
                .map(|groups| Value::Object(sources.vertex_groups(groups))),
            "crease_edge" => Some(sources.keyed_edge_attribute(value)),
            "blender_edge_flags" => value
                .as_array()
                .map(|flags| Value::Array(sources.edge_flags(flags))),
            "blender_attributes" => value.as_array().map(|entries| {
                Value::Array(
                    entries
                        .iter()
                        .map(|entry| sources.attribute_entry(entry))
                        .collect(),
                )
            }),
            "skin_radii" | "skin_roots" => value
                .as_object()
                .map(|values| Value::Object(sources.vertex_object(values, "FLOAT"))),
            "blender_skin_vertices" => value.as_array().map(|vertices| {
                Value::Array(sources.vertex_values(vertices, "FLOAT", || json!({})))
            }),
            ORIGINDEX_ATTRIBUTE => value.as_array().map(|indices| {
                Value::Array(
                    (0..origin.vertices.len())
                        .map(|vertex| {
                            carrier_source(origin, vertex)
                                .and_then(|source| indices.get(source).cloned())
                                .unwrap_or_else(|| json!(ORIGINDEX_NONE))
                        })
                        .collect(),
                )
            }),
            "shade_smooth" => Some(sources.shade_smooth(value)),
            "sharp" => Some(sources.sharp(value)),
            "seam" => Some(sources.seam(value)),
            "blender_polygon_smooth" => value
                .as_array()
                .map(|flags| Value::Array(sources.polygon_smooth(flags))),
            _ => sources
                .graph_attribute(key, value)
                .or_else(|| native_domain(value).map(|_| sources.native_attribute(key, value)))
                .or_else(|| sources.named_attribute(key, value)),
        };
        attributes.insert(key.clone(), remapped.unwrap_or_else(|| value.clone()));
    }
    output.attributes = attributes;
    if let Some(store) = color_store {
        color_layers::write(output, &store);
    }
    if let Some(store) = uv_store {
        uv_layers::write(output, &store);
    }
    if let Some(sources) = code_sources {
        clnor::transfer_custom_normals(
            source,
            output,
            &sources,
            NormalCodeOptions {
                corner_order,
                ..NormalCodeOptions::default()
            },
        )?;
    }
    Ok(())
}

fn corner_offsets(face_sizes: impl Iterator<Item = usize>) -> Vec<usize> {
    let mut offset = 0;
    face_sizes
        .map(|size| {
            let start = offset;
            offset += size;
            start
        })
        .collect()
}

/// The string field `key` of a JSON entry, or `""`.
fn text<'a>(entry: &'a Value, key: &str) -> &'a str {
    entry.get(key).and_then(Value::as_str).unwrap_or_default()
}

/// The `domain` of a named element attribute (one with a `values` object), lowercased.
fn named_domain(value: &Value) -> Option<String> {
    value.get("values")?.as_object()?;
    Some(text(value, "domain").to_ascii_lowercase())
}

/// `entry` with its `values` replaced; the old values array is not cloned.
fn with_values(entry: &Value, values: Value) -> Value {
    let mut fields = Map::new();
    for (key, field) in entry.as_object().into_iter().flatten() {
        if key.as_str() != "values" {
            fields.insert(key.clone(), field.clone());
        }
    }
    fields.insert("values".to_owned(), values);
    Value::Object(fields)
}
fn with_data(entry: &Value, data: Value) -> Value {
    let mut fields = Map::new();
    for (key, field) in entry.as_object().into_iter().flatten() {
        if key.as_str() != "data" {
            fields.insert(key.clone(), field.clone());
        }
    }
    fields.insert("data".to_owned(), data);
    Value::Object(fields)
}

/// Source edges each output edge blends from, with equal weights. Candidates are the distinct input
/// edges joining a source vertex of the output edge's first endpoint to a source vertex of its
/// second. When the endpoints share a source vertex (a subdivision inner edge, for example), a
/// candidate counts only if its two vertices are every source of both endpoints; otherwise the
/// output edge is a new span and gets no source. Blender averages merged duplicate edges, so
/// several candidates blend instead of falling back to default.
fn edge_sources(
    source: &Mesh,
    output: &Mesh,
    vertices: &[Vec<(usize, f64)>],
    output_vertex_index: &HashMap<u32, usize>,
) -> Vec<Vec<(usize, f64)>> {
    let source_vertex_index: HashMap<u32, usize> = source
        .vertices
        .iter()
        .enumerate()
        .map(|(index, vertex)| (vertex.id, index))
        .collect();
    let source_edge_index: HashMap<(usize, usize), usize> = source
        .edges
        .iter()
        .enumerate()
        .filter_map(|(index, edge)| {
            let first = *source_vertex_index.get(&edge.vertices[0])?;
            let second = *source_vertex_index.get(&edge.vertices[1])?;
            Some((edge_key(first, second), index))
        })
        .collect();
    output
        .edges
        .iter()
        .map(|edge| {
            let Some(first) = output_vertex_index
                .get(&edge.vertices[0])
                .and_then(|index| vertices.get(*index))
            else {
                return Vec::new();
            };
            let Some(second) = output_vertex_index
                .get(&edge.vertices[1])
                .and_then(|index| vertices.get(*index))
            else {
                return Vec::new();
            };
            let shared = first
                .iter()
                .any(|(a, _)| second.iter().any(|(b, _)| a == b));
            let endpoint_sources: Vec<usize> = first
                .iter()
                .chain(second)
                .map(|(index, _)| *index)
                .collect();
            let mut found: Vec<usize> = Vec::new();
            for (first_source, _) in first {
                for (second_source, _) in second {
                    let Some(index) =
                        source_edge_index.get(&edge_key(*first_source, *second_source))
                    else {
                        continue;
                    };
                    let spans_all = endpoint_sources
                        .iter()
                        .all(|source| source == first_source || source == second_source);
                    if (!shared || spans_all) && !found.contains(index) {
                        found.push(*index);
                    }
                }
            }
            let share = equal_share(found.len());
            found.into_iter().map(|index| (index, share)).collect()
        })
        .collect()
}

/// Source corners each output corner blends from. An output corner of output face `f` (source face
/// `sf`) on output vertex `v` takes the input vertices of `v` that belong to `sf`, weighted by their
/// origin weights renormalised over those vertices, each mapped to its corner in `sf`. A corner with
/// no such vertex, or on a generated face, has no sources and takes the domain default.
pub(crate) fn corner_sources(
    source: &Mesh,
    origin: &Origin,
    output: &Mesh,
    output_vertex_index: &HashMap<u32, usize>,
) -> Vec<Vec<(usize, f64)>> {
    let offsets = corner_offsets(source.faces.iter().map(|face| face.vertices.len()));
    // Every source corner keyed by its face and vertex, built once per member, so an output corner
    // finds its source corner by lookup instead of scanning its source face.
    let mut source_corners: HashMap<(usize, u32), usize> = HashMap::new();
    for (index, face) in source.faces.iter().enumerate() {
        for (position, id) in face.vertices.iter().enumerate() {
            source_corners
                .entry((index, *id))
                .or_insert(offsets[index] + position);
        }
    }
    let mut corners = Vec::new();
    for (face, source_face) in output.faces.iter().zip(&origin.faces) {
        for vertex_id in &face.vertices {
            let vertex_sources = output_vertex_index
                .get(vertex_id)
                .and_then(|index| origin.vertices.get(*index));
            let mut kept = Vec::new();
            if let (Some(index), Some(vertex_sources)) = (*source_face, vertex_sources) {
                for (vertex, weight) in vertex_sources {
                    let Some(id) = source.vertices.get(*vertex).map(|vertex| vertex.id) else {
                        continue;
                    };
                    if let Some(&corner) = source_corners.get(&(index, id)) {
                        kept.push((corner, *weight));
                    }
                }
            }
            let total: f64 = kept.iter().map(|(_, weight)| *weight).sum();
            if total > 0.0 {
                for (_, weight) in &mut kept {
                    *weight /= total;
                }
            } else {
                kept.clear();
            }
            corners.push(kept);
        }
    }
    corners
}

/// Corner sources by Blender's `interp_weights_poly_v2`. New output corners are projected into the
/// source face's float plane basis and snap to a source point/edge within Blender's scale-aware epsilon.
pub(crate) fn mean_value_corners<'a>(
    source: &Mesh,
    points: &[DVec3],
    faces: impl IntoIterator<Item = (&'a [usize], Option<usize>)>,
) -> Vec<Vec<(usize, f64)>> {
    let offsets = corner_offsets(source.faces.iter().map(|face| face.vertices.len()));
    let source_index: HashMap<u32, usize> = source
        .vertices
        .iter()
        .enumerate()
        .map(|(index, vertex)| (vertex.id, index))
        .collect();
    let mut corners = Vec::new();
    for (vertices, source_face) in faces {
        let Some(index) = source_face.filter(|index| *index < source.faces.len()) else {
            corners.extend(vertices.iter().map(|_| Vec::new()));
            continue;
        };
        let polygon: Option<Vec<DVec3>> = source.faces[index]
            .vertices
            .iter()
            .map(|id| {
                source_index
                    .get(id)
                    .map(|position| source.vertices[*position].co)
            })
            .collect();
        let projected = polygon.as_deref().and_then(project_polygon);
        // One weight buffer per source face, reused for each of its output vertices.
        let mut weights = projected
            .as_ref()
            .map_or_else(Vec::new, |(polygon, _)| vec![0.0_f32; polygon.len()]);
        for &vertex in vertices {
            let interpolated = match (projected.as_ref(), points.get(vertex)) {
                (Some((polygon, axes)), Some(point)) => {
                    interp_weights_poly_v2(&mut weights, polygon, project_point(*point, *axes))
                }
                _ => false,
            };
            corners.push(if interpolated {
                weights
                    .iter()
                    .enumerate()
                    .filter(|(_, weight)| **weight != 0.0)
                    .map(|(corner, weight)| (offsets[index] + corner, f64::from(*weight)))
                    .collect()
            } else {
                Vec::new()
            });
        }
    }
    corners
}

type ProjectionAxes = ([f32; 3], [f32; 3]);

fn sub3_f32(first: [f32; 3], second: [f32; 3]) -> [f32; 3] {
    [
        first[0] - second[0],
        first[1] - second[1],
        first[2] - second[2],
    ]
}

fn cross3_f32(first: [f32; 3], second: [f32; 3]) -> [f32; 3] {
    [
        first[1] * second[2] - first[2] * second[1],
        first[2] * second[0] - first[0] * second[2],
        first[0] * second[1] - first[1] * second[0],
    ]
}
/// Blender's `face_normal_calc` (triangle, quad, or Newell) and dominant-axis projection, in f32.
fn project_polygon(polygon: &[DVec3]) -> Option<(Vec<[f32; 2]>, ProjectionAxes)> {
    if polygon.len() < 3 {
        return None;
    }
    let points: Vec<[f32; 3]> = polygon
        .iter()
        .map(|point| [point.x as f32, point.y as f32, point.z as f32])
        .collect();
    let mut normal = match points.as_slice() {
        [first, second, third] => cross3_f32(sub3_f32(*first, *second), sub3_f32(*second, *third)),
        [first, second, third, fourth] => {
            cross3_f32(sub3_f32(*first, *third), sub3_f32(*second, *fourth))
        }
        _ => {
            let mut normal = [0.0_f32; 3];
            let mut previous = points[points.len() - 1];
            for current in &points {
                normal[0] += (previous[1] - current[1]) * (previous[2] + current[2]);
                normal[1] += (previous[2] - current[2]) * (previous[0] + current[0]);
                normal[2] += (previous[0] - current[0]) * (previous[1] + current[1]);
                previous = *current;
            }
            normal
        }
    };
    let length_squared = dot3_f32(normal, normal);
    if length_squared > 1.0e-35 {
        let inverse_length = 1.0 / length_squared.sqrt();
        normal = normal.map(|component| component * inverse_length);
    } else {
        normal = [0.0, 0.0, 1.0];
    }
    let axes = orthographic_projection_axes(normal);
    let projected = points
        .iter()
        .map(|&point| project_point_f32(point, axes))
        .collect();
    Some((projected, axes))
}

fn orthographic_projection_axes(normal: [f32; 3]) -> ProjectionAxes {
    let length_squared = normal[0] * normal[0] + normal[1] * normal[1];
    if length_squared > f32::EPSILON {
        let inverse_length = 1.0 / length_squared.sqrt();
        let first = [normal[1] * inverse_length, -normal[0] * inverse_length, 0.0];
        let second = [
            -normal[2] * first[1],
            normal[2] * first[0],
            normal[0] * first[1] - normal[1] * first[0],
        ];
        (first, second)
    } else {
        (
            [if normal[2] < 0.0 { -1.0 } else { 1.0 }, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        )
    }
}

fn project_point(point: DVec3, axes: ProjectionAxes) -> [f32; 2] {
    project_point_f32([point.x as f32, point.y as f32, point.z as f32], axes)
}

fn project_point_f32(point: [f32; 3], axes: ProjectionAxes) -> [f32; 2] {
    [dot3_f32(axes.0, point), dot3_f32(axes.1, point)]
}

fn dot3_f32(first: [f32; 3], second: [f32; 3]) -> f32 {
    first[0] * second[0] + first[1] * second[1] + first[2] * second[2]
}

#[expect(
    clippy::cast_precision_loss,
    reason = "source element counts are far below 2^53"
)]
fn equal_share(count: usize) -> f64 {
    1.0 / count as f64
}

/// Keyed values as `(index, value)` pairs: keys are `{prefix}{index}` in canonical decimal form.
fn keyed_values<'a, T>(values: &'a Map<String, Value>, prefix: &str) -> HashMap<T, &'a Value>
where
    T: FromStr + Display + Hash + Eq,
{
    values
        .iter()
        .filter_map(|(key, value)| {
            let index: T = key.strip_prefix(prefix)?.parse().ok()?;
            (format!("{prefix}{index}") == *key).then_some((index, value))
        })
        .collect()
}

struct Sources<'a> {
    source: &'a Mesh,
    output: &'a Mesh,
    origin: &'a Origin,
    staged: Option<&'a Staged>,
    point_identity: Option<&'a [usize]>,
    edge_origin: &'a [Vec<(usize, f64)>],
    corner_origin: &'a [Vec<(usize, f64)>],
    weighted_corner_booleans: bool,
    color_default: [f64; 4],
}

impl Sources<'_> {
    /// How output vertex `index` mixes: its staged or pair mixing when it has one, otherwise its flat sources.
    fn vertex_mix(&self, index: usize) -> Mix<'_> {
        let mixing = self
            .staged
            .and_then(|staged| staged.vertices.get(index)?.as_ref());
        element_mix(mixing, &self.origin.vertices[index])
    }

    /// How output corner `index` mixes over `origins` (the shared corner sources, or a FLOAT2 layer's own):
    /// its stages when it has them, otherwise its flat sources.
    fn corner_mix<'b>(&'b self, origins: &'b [Vec<(usize, f64)>], index: usize) -> Mix<'b> {
        match self
            .staged
            .and_then(|staged| staged.corners.as_ref()?.get(index))
        {
            Some(stages) => Mix::Staged(stages),
            None => Mix::Flat(&origins[index]),
        }
    }

    /// [`blend_mix`] of output corner `corner` over `origins`. A quaternion corner that Bevel interpolated
    /// mixes through the expmap even from one full-weight source, as Blender's interpolation does; other
    /// corners blend as [`blend_mix`] does.
    fn blend_corner<'a>(
        &self,
        origins: &[Vec<(usize, f64)>],
        corner: usize,
        value: impl Fn(usize) -> Option<&'a Value>,
        data_type: &str,
        weighted_booleans: bool,
    ) -> Option<Value> {
        let mix = self.corner_mix(origins, corner);
        if let Mix::Flat(sources) = mix
            && data_type == "QUATERNION"
            && self
                .origin
                .corner_interpolated
                .as_ref()
                .is_some_and(|flags| flags.get(corner) == Some(&true))
        {
            return blend_quaternion(present_sources(&value, sources).into_iter());
        }
        blend_mix(value, mix, data_type, weighted_booleans)
    }

    /// The corner origins of a FLOAT2 layer. Blender merges each layer's sources independently, so a layer
    /// listed in `corner_float2_sources` uses its own origins; other data uses the shared corner origins.
    fn corner_origins_for_layer(
        &self,
        data_type: &str,
        key: &CornerFloat2LayerKey,
    ) -> &[Vec<(usize, f64)>] {
        if !data_type.eq_ignore_ascii_case("FLOAT2") {
            return self.corner_origin;
        }
        self.origin
            .corner_float2_sources
            .as_deref()
            .and_then(|layers| layers.iter().find(|layer| &layer.key == key))
            .map_or(self.corner_origin, |layer| layer.corners.as_slice())
    }

    /// The FLOAT2 layer key of a `blender_attributes` corner entry: the `uv_layers` layer of that name when
    /// one exists.
    fn blender_attribute_corner_key(&self, name: &str) -> CornerFloat2LayerKey {
        if uv_layers::read(self.source).index_of(name).is_some() {
            CornerFloat2LayerKey::Uv(name.to_owned())
        } else {
            CornerFloat2LayerKey::BlenderAttribute(name.to_owned())
        }
    }

    /// Whether point data of `data_type` replays Bevel's ordered interpolation events, rather than blending origins.
    fn replays_points(&self, data_type: &str) -> bool {
        data_type == "QUATERNION" && self.origin.point_interpolation_events.is_some()
    }

    /// Point quaternions replayed from Bevel's ordered callbacks. Each event mixes its sources' current values,
    /// and a mutated source keeps the written value for later events. An output vertex takes its identity's
    /// stored value unless an event wrote it, so a vertex without one has no entry. `None` without an event
    /// stream or point identities.
    fn point_quaternion_values(&self, stored: &[Option<&Value>]) -> Option<Vec<Option<Value>>> {
        let events = self.origin.point_interpolation_events.as_ref()?;
        let identities = self.point_identity?;
        let mut source_values: Vec<Value> = stored
            .iter()
            .map(|value| (*value).map_or_else(|| default_value("QUATERNION"), Value::clone))
            .collect();
        let mut output_values: Vec<Option<Value>> = identities
            .iter()
            .map(|source| stored.get(*source).copied().flatten().cloned())
            .collect();
        for event in events {
            let value = blend_quaternion(event.sources.iter().filter_map(|(source, weight)| {
                source_values.get(*source).map(|value| (value, *weight))
            }))
            .unwrap_or_else(|| default_value("QUATERNION"));
            if let Some(output) = output_values.get_mut(event.output_vertex) {
                *output = Some(value.clone());
            }
            if let Some(source_vertex) = event.mutated_source_vertex
                && let Some(source_value) = source_values.get_mut(source_vertex)
            {
                *source_value = value;
            }
        }
        Some(output_values)
    }

    /// Interpolated vertex-group weights keyed `v{output id}` under each group name.
    fn vertex_groups(&self, groups: &Map<String, Value>) -> Map<String, Value> {
        groups
            .iter()
            .map(|(name, weights)| {
                let by_id: HashMap<u32, f64> = weights
                    .as_object()
                    .into_iter()
                    .flatten()
                    .filter_map(|(key, value)| {
                        Some((key.strip_prefix('v')?.parse().ok()?, value.as_f64()?))
                    })
                    .collect();
                let mut output = Map::new();
                for (vertex, sources) in self.origin.vertices.iter().enumerate() {
                    let mut found = false;
                    let mut sum = 0.0;
                    for (index, weight) in sources {
                        if let Some(value) = by_id.get(&self.source.vertices[*index].id) {
                            found = true;
                            sum += weight * value;
                        }
                    }
                    if found {
                        output.insert(format!("v{}", self.output.vertices[vertex].id), json!(sum));
                    }
                }
                (name.clone(), Value::Object(output))
            })
            .collect()
    }

    /// An edge-domain float map (`{"domain","type","values":{"e{id}":v}}`) moved to output edge IDs,
    /// each blended over its source edges (see [`blend`]).
    fn keyed_edge_attribute(&self, attribute: &Value) -> Value {
        let by_id: HashMap<u32, &Value> = attribute
            .get("values")
            .and_then(Value::as_object)
            .map(|values| keyed_values(values, "e"))
            .unwrap_or_default();
        let mut output = Map::new();
        for (edge, sources) in self.output.edges.iter().zip(self.edge_origin) {
            let blended = blend(
                sources.iter().filter_map(|(index, weight)| {
                    let id = self.source.edges.get(*index)?.id;
                    by_id.get(&id).map(|value| (*value, *weight))
                }),
                "FLOAT",
            );
            if let Some(value) = blended {
                output.insert(format!("e{}", edge.id), value);
            }
        }
        with_values(attribute, Value::Object(output))
    }

    /// Edge values blend over source edges except for direct transform-matrix copies.
    fn edge_values(
        &self,
        values: &[Value],
        data_type: &str,
        default: impl Fn() -> Value,
    ) -> Vec<Value> {
        self.edge_origin
            .iter()
            .map(|sources| {
                if is_float4x4(data_type)
                    && let [(source, weight)] = sources.as_slice()
                    && weight.to_bits() == 1.0_f64.to_bits()
                {
                    values.get(*source).cloned().unwrap_or_else(&default)
                } else {
                    blend_indexed(values, sources, data_type).unwrap_or_else(&default)
                }
            })
            .collect()
    }

    /// Legacy Blender edge flags combined from every source edge.
    fn edge_flags(&self, flags: &[Value]) -> Vec<Value> {
        self.edge_origin
            .iter()
            .map(|sources| {
                let flag = |name: &str| {
                    sources.iter().any(|(index, weight)| {
                        *weight > 0.0
                            && flags
                                .get(*index)
                                .and_then(|flags| flags.get(name))
                                .and_then(Value::as_bool)
                                == Some(true)
                    })
                };
                json!({"sharp": flag("sharp"), "seam": flag("seam")})
            })
            .collect()
    }

    /// Canonical face smooth flags copied from each output face's source face.
    fn shade_smooth(&self, attribute: &Value) -> Value {
        let Some(values) = attribute.get("values").and_then(Value::as_object) else {
            return attribute.clone();
        };
        let remapped: Map<String, Value> = self
            .output
            .faces
            .iter()
            .zip(&self.origin.faces)
            .map(|(face, source)| {
                let smooth = source
                    .and_then(|index| self.source.faces.get(index))
                    .and_then(|source_face| values.get(&format!("f{}", source_face.id)))
                    .and_then(Value::as_bool)
                    .unwrap_or(false);
                (format!("f{}", face.id), json!(smooth))
            })
            .collect();
        with_values(attribute, Value::Object(remapped))
    }

    /// Canonical sharp flags OR-ed over source edges, as Blender's edge-merge path does.
    fn sharp(&self, attribute: &Value) -> Value {
        self.edge_boolean_attribute(attribute, "sharp")
    }

    /// Native seam flags use the same edge-merge propagation as sharp flags.
    fn seam(&self, attribute: &Value) -> Value {
        self.edge_boolean_attribute(attribute, "seam")
    }

    fn edge_boolean_attribute(&self, attribute: &Value, name: &str) -> Value {
        if attribute.get("values").and_then(Value::as_object).is_none() {
            return attribute.clone();
        }
        let source_flags = edge_flags(self.source, name);
        // Sparse, as the source attribute is: an output edge carries an entry only when a source edge
        // with the flag set contributes to it, and absence reads as false.
        let remapped: Map<String, Value> = self
            .output
            .edges
            .iter()
            .zip(self.edge_origin)
            .filter(|(_, sources)| {
                sources
                    .iter()
                    .any(|(index, weight)| *weight > 0.0 && source_flags.get(*index) == Some(&true))
            })
            .map(|(edge, _)| (format!("e{}", edge.id), json!(true)))
            .collect();
        with_values(attribute, Value::Object(remapped))
    }

    /// Legacy polygon-smooth flags copied from each output face's source face.
    fn polygon_smooth(&self, flags: &[Value]) -> Vec<Value> {
        self.origin
            .faces
            .iter()
            .map(|source| {
                source
                    .and_then(|index| flags.get(index))
                    .cloned()
                    .unwrap_or(json!(false))
            })
            .collect()
    }

    /// Blender's color data type of the color attribute `name`: the `blender_metadata` entry written
    /// with it, or `FLOAT_COLOR` when there is none.
    fn color_data_type(&self, name: &str) -> &'static str {
        let stored = self
            .source
            .attributes
            .get("blender_metadata")
            .and_then(|metadata| metadata.get("color_data_types"))
            .and_then(Value::as_object)
            .and_then(|types| types.get(name))
            .and_then(Value::as_str);
        if stored == Some("BYTE_COLOR") {
            "BYTE_COLOR"
        } else {
            "FLOAT_COLOR"
        }
    }

    /// A native element attribute (`{"domain", "type", "values"}`) moved to the output elements of its
    /// domain. Point, edge, and corner values blend over their origins by weight, and a face value copies
    /// its source face's value. An output element with no source value gets no entry, so a reader takes
    /// its type's default for the missing key and no default is ever written. A corner row is written
    /// only when one of its corners blends a value; its other corners take the type's default. Booleans
    /// use Blender's `DefaultMixer<bool>` weighted threshold, and colors use the mixer for their source encoding,
    /// selected by the `blender_metadata` color type.
    fn native_attribute(&self, key: &str, attribute: &Value) -> Value {
        let Some(values) = attribute.get("values").and_then(Value::as_object) else {
            return attribute.clone();
        };
        let kind = text(attribute, "type");
        let data_type = if kind == "color" {
            attribute_keys::user_name(key).map_or("FLOAT_COLOR", |name| self.color_data_type(name))
        } else {
            blend_type(kind)
        };
        let default = || default_value(data_type);
        let remapped: Map<String, Value> = match text(attribute, "domain") {
            "point" => {
                let stored: Vec<Option<&Value>> = self
                    .source
                    .vertices
                    .iter()
                    .map(|vertex| values.get(&format!("v{}", vertex.id)))
                    .collect();
                let replayed = self
                    .replays_points(data_type)
                    .then(|| self.point_quaternion_values(&stored))
                    .flatten();
                let lookup = |source: usize| stored.get(source).copied().flatten();
                self.output
                    .vertices
                    .iter()
                    .enumerate()
                    .filter_map(|(index, vertex)| {
                        let value = match &replayed {
                            Some(replay) => replay.get(index).cloned().flatten()?,
                            None => blend_mix(lookup, self.vertex_mix(index), data_type, false)?,
                        };
                        Some((format!("v{}", vertex.id), value))
                    })
                    .collect()
            }
            "edge" => {
                let stored: Vec<Option<&Value>> = self
                    .source
                    .edges
                    .iter()
                    .map(|edge| values.get(&format!("e{}", edge.id)))
                    .collect();
                self.output
                    .edges
                    .iter()
                    .zip(self.edge_origin)
                    .filter_map(|(edge, sources)| {
                        let value = blend_sources(&stored, sources, data_type)?;
                        Some((format!("e{}", edge.id), value))
                    })
                    .collect()
            }
            "face" => self
                .output
                .faces
                .iter()
                .zip(&self.origin.faces)
                .filter_map(|(face, source)| {
                    let value = source
                        .and_then(|index| self.source.faces.get(index))
                        .and_then(|source_face| values.get(&format!("f{}", source_face.id)))?;
                    Some((format!("f{}", face.id), value.clone()))
                })
                .collect(),
            "corner" => {
                let corner_origins = self.corner_origins_for_layer(
                    data_type,
                    &CornerFloat2LayerKey::Attribute(key.to_owned()),
                );
                let corner_count: usize = self
                    .source
                    .faces
                    .iter()
                    .map(|face| face.vertices.len())
                    .sum();
                let mut stored: Vec<Option<&Value>> = vec![None; corner_count];
                let offsets =
                    corner_offsets(self.source.faces.iter().map(|face| face.vertices.len()));
                for (face, offset) in self.source.faces.iter().zip(offsets) {
                    let Some(row) = values
                        .get(&format!("f{}", face.id))
                        .and_then(Value::as_array)
                    else {
                        continue;
                    };
                    for (corner, value) in row.iter().enumerate().take(face.vertices.len()) {
                        stored[offset + corner] = Some(value);
                    }
                }
                let offsets =
                    corner_offsets(self.output.faces.iter().map(|face| face.vertices.len()));
                let lookup = |source: usize| stored.get(source).copied().flatten();
                self.output
                    .faces
                    .iter()
                    .zip(offsets)
                    .filter_map(|(face, offset)| {
                        let blended: Vec<Option<Value>> = (offset..offset + face.vertices.len())
                            .map(|corner| {
                                self.blend_corner(
                                    corner_origins,
                                    corner,
                                    lookup,
                                    data_type,
                                    self.weighted_corner_booleans,
                                )
                            })
                            .collect();
                        if blended.iter().all(Option::is_none) {
                            return None;
                        }
                        let row = blended
                            .into_iter()
                            .map(|value| value.unwrap_or_else(default))
                            .collect();
                        Some((format!("f{}", face.id), Value::Array(row)))
                    })
                    .collect()
            }
            _ => return attribute.clone(),
        };
        with_values(attribute, Value::Object(remapped))
    }

    /// Positional vertex values, blended by origin weights (see [`blend`]).
    fn vertex_values(
        &self,
        values: &[Value],
        data_type: &str,
        default: impl Fn() -> Value,
    ) -> Vec<Value> {
        if self.replays_points(data_type) {
            let stored: Vec<Option<&Value>> = (0..self.source.vertices.len())
                .map(|index| values.get(index))
                .collect();
            if let Some(replayed) = self.point_quaternion_values(&stored) {
                return replayed
                    .into_iter()
                    .map(|value| value.unwrap_or_else(&default))
                    .collect();
            }
        }
        (0..self.origin.vertices.len())
            .map(|index| {
                blend_mix(
                    |source| values.get(source),
                    self.vertex_mix(index),
                    data_type,
                    false,
                )
                .unwrap_or_else(&default)
            })
            .collect()
    }

    /// Keyed vertex values (`"{index}"` keys, storage order) moved to output indices.
    fn vertex_object(&self, values: &Map<String, Value>, data_type: &str) -> Map<String, Value> {
        let by_index: HashMap<usize, &Value> = keyed_values(values, "");
        let mut output = Map::new();
        for (index, sources) in self.origin.vertices.iter().enumerate() {
            let blended = blend(
                sources.iter().filter_map(|(source, weight)| {
                    by_index.get(source).map(|value| (*value, *weight))
                }),
                data_type,
            );
            if let Some(value) = blended {
                output.insert(index.to_string(), value);
            }
        }
        output
    }

    /// One `blender_attributes` entry; POINT, EDGE, FACE, and CORNER entries are remapped and
    /// unknown domains are returned unchanged.
    fn attribute_entry(&self, entry: &Value) -> Value {
        let Some(values) = entry.get("values").and_then(Value::as_array) else {
            return entry.clone();
        };
        let Some(remapped) = self.remap_attribute_values(
            text(entry, "domain"),
            text(entry, "data_type"),
            text(entry, "name"),
            values,
        ) else {
            return entry.clone();
        };
        with_values(entry, Value::Array(remapped))
    }

    fn graph_attribute(&self, name: &str, attribute: &Value) -> Option<Value> {
        let values = attribute.get("data")?.as_array()?;
        let remapped = self.remap_attribute_values(
            text(attribute, "domain"),
            text(attribute, "data_type"),
            name,
            values,
        )?;
        Some(with_data(attribute, Value::Array(remapped)))
    }

    fn remap_attribute_values(
        &self,
        domain: &str,
        data_type: &str,
        name: &str,
        values: &[Value],
    ) -> Option<Vec<Value>> {
        let default = || default_value(data_type);
        Some(match domain {
            "POINT" => self.vertex_values(values, data_type, default),
            "EDGE" => self.edge_values(values, data_type, default),
            "FACE" => self
                .origin
                .faces
                .iter()
                .map(|source| {
                    source
                        .and_then(|index| values.get(index).cloned())
                        .unwrap_or_else(default)
                })
                .collect(),
            "CORNER" => {
                let corner_origins = if data_type.eq_ignore_ascii_case("FLOAT2") {
                    let key = self.blender_attribute_corner_key(name);
                    self.corner_origins_for_layer(data_type, &key)
                } else {
                    self.corner_origin
                };
                (0..corner_origins.len())
                    .map(|corner| {
                        self.blend_corner(
                            corner_origins,
                            corner,
                            |source| values.get(source),
                            data_type,
                            self.weighted_corner_booleans,
                        )
                        .unwrap_or_else(&default)
                    })
                    .collect()
            }
            _ => return None,
        })
    }
    /// One canonical color layer rebuilt with the Blender mixer named in the source metadata.
    fn color_layer(&self, layer: &ColorLayer) -> ColorLayer {
        let data_type = self.color_data_type(&layer.name);
        let mix = |sources: &[(usize, f64)]| {
            if data_type == "BYTE_COLOR" {
                if sources.is_empty() {
                    self.color_default
                } else {
                    color_layers::mix_byte_color(&layer.values, sources)
                }
            } else {
                blend_rgba(
                    sources
                        .iter()
                        .filter_map(|(index, weight)| Some((*layer.values.get(*index)?, *weight))),
                    data_type,
                )
                .unwrap_or(self.color_default)
            }
        };
        let values = match layer.domain {
            ColorDomain::Point => self
                .origin
                .vertices
                .iter()
                .map(|sources| mix(sources))
                .collect(),
            ColorDomain::Face => self
                .origin
                .faces
                .iter()
                .map(|source| source.map_or(self.color_default, |source| mix(&[(source, 1.0)])))
                .collect(),
            ColorDomain::Corner => self
                .corner_origin
                .iter()
                .map(|sources| mix(sources))
                .collect(),
        };
        ColorLayer {
            name: layer.name.clone(),
            domain: layer.domain,
            values,
        }
    }
    /// One UV layer rebuilt for the output faces. Each output corner blends the UVs of its source
    /// corners by weight. A source corner whose face has no entry reads as `[0, 1]` and keeps its
    /// weight (no renormalisation). An output face gets an entry when any of its corners has a source
    /// with an entry, so a modifier that leaves face sources empty (screw) keeps its UVs. A corner's
    /// pin is the pin of its heaviest source corner (see [`heaviest_source`]); `pinned` is omitted
    /// when no contributing source corner has pins.
    fn uv_layer(&self, layer: &UvLayer) -> UvLayer {
        let source_offsets =
            corner_offsets(self.source.faces.iter().map(|face| face.vertices.len()));
        let output_offsets =
            corner_offsets(self.output.faces.iter().map(|face| face.vertices.len()));
        let source_corners: usize = self
            .source
            .faces
            .iter()
            .map(|face| face.vertices.len())
            .sum();
        let mut flat = vec![[0.0, 1.0]; source_corners];
        // Pin per source corner; `None` where the source face has no `pinned` array.
        let mut flat_pinned: Vec<Option<bool>> = vec![None; source_corners];
        // Source corners whose face has an entry for this layer.
        let mut filled = vec![false; source_corners];
        for (index, &at) in source_offsets.iter().enumerate() {
            let Some(Some(uv)) = layer.faces.get(index) else {
                continue;
            };
            for (corner, value) in uv.iter().enumerate() {
                flat[at + corner] = *value;
                filled[at + corner] = true;
            }
            if let Some(Some(pins)) = layer.pinned.get(index) {
                for (corner, pin) in pins.iter().enumerate() {
                    flat_pinned[at + corner] = Some(*pin);
                }
            }
        }
        let mut faces: Vec<Option<Vec<[f64; 2]>>> = Vec::with_capacity(self.output.faces.len());
        let mut pinned: Vec<Option<Vec<bool>>> = Vec::with_capacity(self.output.faces.len());
        let uv_corner_origins =
            self.corner_origins_for_layer("FLOAT2", &CornerFloat2LayerKey::Uv(layer.name.clone()));
        for (face_index, face) in self.output.faces.iter().enumerate() {
            let corners = &self.corner_origin[output_offsets[face_index]..][..face.vertices.len()];
            let uv_corners =
                &uv_corner_origins[output_offsets[face_index]..][..face.vertices.len()];
            if !uv_corners
                .iter()
                .flatten()
                .any(|(source, _)| filled[*source])
            {
                faces.push(None);
                pinned.push(None);
                continue;
            }
            let uv: Vec<[f64; 2]> = uv_corners
                .iter()
                .map(|sources| blend_uv(&flat, sources))
                .collect();
            faces.push(Some(uv));
            let has_pins = corners
                .iter()
                .flatten()
                .any(|(source, _)| flat_pinned.get(*source).is_some_and(Option::is_some));
            if has_pins {
                let pins: Vec<bool> = corners
                    .iter()
                    .map(|sources| {
                        heaviest_source(sources)
                            .and_then(|source| flat_pinned.get(source).copied().flatten())
                            == Some(true)
                    })
                    .collect();
                pinned.push(Some(pins));
            } else {
                pinned.push(None);
            }
        }
        UvLayer {
            name: layer.name.clone(),
            faces,
            pinned,
        }
    }

    /// A named attribute keyed by element ID (`{"domain","type","values":{prefix+id: value}}`), moved to
    /// the output's IDs. Point and edge values blend over their source elements, a face value copies its
    /// source face's, and corner values (one array per face) blend over the source corners. Other domains
    /// are copied unchanged by the caller.
    fn named_attribute(&self, key: &str, attribute: &Value) -> Option<Value> {
        let domain = named_domain(attribute)?;
        let kind = text(attribute, "type").to_ascii_uppercase();
        let values = attribute.get("values").and_then(Value::as_object)?;
        let mut output = Map::new();
        match domain.as_str() {
            "point" => {
                let by_id: HashMap<u32, &Value> = keyed_values(values, "v");
                let replayed = self
                    .replays_points(&kind)
                    .then(|| {
                        let stored: Vec<Option<&Value>> = self
                            .source
                            .vertices
                            .iter()
                            .map(|vertex| by_id.get(&vertex.id).copied())
                            .collect();
                        self.point_quaternion_values(&stored)
                    })
                    .flatten();
                for (index, vertex) in self.output.vertices.iter().enumerate() {
                    let lookup = |source: usize| {
                        let id = self.source.vertices.get(source)?.id;
                        by_id.get(&id).copied()
                    };
                    let value = match &replayed {
                        Some(replay) => replay.get(index).cloned().flatten(),
                        None => blend_mix(lookup, self.vertex_mix(index), &kind, false),
                    };
                    if let Some(value) = value {
                        output.insert(format!("v{}", vertex.id), value);
                    }
                }
            }
            "edge" => {
                let by_id: HashMap<u32, &Value> = keyed_values(values, "e");
                for (edge, sources) in self.output.edges.iter().zip(self.edge_origin) {
                    let blended = blend(
                        sources.iter().filter_map(|(index, weight)| {
                            let id = self.source.edges.get(*index)?.id;
                            by_id.get(&id).map(|value| (*value, *weight))
                        }),
                        &kind,
                    );
                    if let Some(value) = blended {
                        output.insert(format!("e{}", edge.id), value);
                    }
                }
            }
            "face" => {
                for (face, source) in self.output.faces.iter().zip(&self.origin.faces) {
                    let value = source
                        .and_then(|index| self.source.faces.get(index))
                        .and_then(|source| values.get(&format!("f{}", source.id)));
                    if let Some(value) = value {
                        output.insert(format!("f{}", face.id), value.clone());
                    }
                }
            }
            "corner" => {
                let corner_origins = self.corner_origins_for_layer(
                    &kind,
                    &CornerFloat2LayerKey::Attribute(key.to_owned()),
                );
                let source_offsets =
                    corner_offsets(self.source.faces.iter().map(|face| face.vertices.len()));
                let source_corners: usize = self
                    .source
                    .faces
                    .iter()
                    .map(|face| face.vertices.len())
                    .sum();
                let mut authored: Vec<Option<&Value>> = vec![None; source_corners];
                for (index, face) in self.source.faces.iter().enumerate() {
                    let Some(corners) = values
                        .get(&format!("f{}", face.id))
                        .and_then(Value::as_array)
                    else {
                        continue;
                    };
                    for (corner, value) in corners.iter().enumerate().take(face.vertices.len()) {
                        authored[source_offsets[index] + corner] = Some(value);
                    }
                }
                let output_offsets =
                    corner_offsets(self.output.faces.iter().map(|face| face.vertices.len()));
                for (face_index, face) in self.output.faces.iter().enumerate() {
                    let first = output_offsets[face_index];
                    let lookup = |source: usize| authored.get(source).copied().flatten();
                    let row: Vec<Option<Value>> = (first..first + face.vertices.len())
                        .map(|corner| {
                            self.blend_corner(
                                corner_origins,
                                corner,
                                lookup,
                                &kind,
                                self.weighted_corner_booleans,
                            )
                        })
                        .collect();
                    if row.iter().all(Option::is_none) {
                        continue;
                    }
                    let row = row
                        .into_iter()
                        .map(|value| value.unwrap_or_else(|| default_value(&kind)))
                        .collect();
                    output.insert(format!("f{}", face.id), Value::Array(row));
                }
            }
            _ => return None,
        }
        Some(with_values(attribute, Value::Object(output)))
    }
}

/// Index of the heaviest `(index, weight)` source; ties go to the lowest index.
fn heaviest_source(sources: &[(usize, f64)]) -> Option<usize> {
    sources
        .iter()
        .max_by(|left, right| left.1.total_cmp(&right.1).then(right.0.cmp(&left.0)))
        .map(|(index, _)| *index)
}

/// [`blend`] over `values` at the `(index, weight)` sources; indices past the end are skipped.
fn blend_indexed(values: &[Value], sources: &[(usize, f64)], data_type: &str) -> Option<Value> {
    blend_mix(
        |index| values.get(index),
        Mix::Flat(sources),
        data_type,
        false,
    )
}

/// The `(value, weight)` pairs of the sources that have a value.
fn present_sources<'a>(
    value: &impl Fn(usize) -> Option<&'a Value>,
    sources: &[(usize, f64)],
) -> Vec<(&'a Value, f64)> {
    sources
        .iter()
        .filter_map(|&(index, weight)| Some((value(index)?, weight)))
        .collect()
}

/// Blender's mix of `items`: two quaternion sources with a factor are `mix2` (`math::interpolate`); any
/// other set is the weighted `mix_indices` of [`blend`].
fn mix2_or_blend(
    items: Vec<(&Value, f64)>,
    data_type: &str,
    weighted_booleans: bool,
) -> Option<Value> {
    if let [(first, _), (second, factor)] = items.as_slice()
        && data_type == "QUATERNION"
        && is_quaternion(first)
        && is_quaternion(second)
    {
        return quaternion_mix2(first, second, *factor as f32);
    }
    blend_with_boolean_mixer(items.into_iter(), data_type, weighted_booleans)
}

/// [`blend`] of an element's mix. Flat sources blend once. Staged sources blend each stage first, then the
/// stage values by the stage weights, as Blender's `mix_attrs` does for face-sampled elements. A stage whose
/// sources have no values is skipped, as flat sources without values are.
fn blend_mix<'a>(
    value: impl Fn(usize) -> Option<&'a Value>,
    mix: Mix<'_>,
    data_type: &str,
    weighted_booleans: bool,
) -> Option<Value> {
    match mix {
        Mix::Pair(sources) => mix2_or_blend(
            present_sources(&value, sources),
            data_type,
            weighted_booleans,
        ),
        Mix::Flat(sources) => blend_with_boolean_mixer(
            sources
                .iter()
                .filter_map(|&(index, weight)| Some((value(index)?, weight))),
            data_type,
            weighted_booleans,
        ),
        Mix::Staged(stages) => {
            // A stage of two sources is Blender's `mix2` (an edge midpoint of the slots), while a stage of more
            // sources is the weighted `mix_indices` of the centre slot.
            let mixed: Vec<(Value, f64)> = stages
                .iter()
                .filter_map(|stage| {
                    let inner = mix2_or_blend(
                        present_sources(&value, &stage.sources),
                        data_type,
                        weighted_booleans,
                    )?;
                    Some((inner, stage.weight))
                })
                .collect();
            blend_outer(&mixed, data_type, weighted_booleans)
        }
    }
}

/// The outer mix of staged values. Blender always mixes a face-sampled element's slots (`mix4`), even a
/// single full-weight slot, so quaternions take the exponential-map mix here, unlike a copied element.
fn blend_outer(items: &[(Value, f64)], data_type: &str, weighted_booleans: bool) -> Option<Value> {
    if items.is_empty() {
        return None;
    }
    let borrowed: Vec<(&Value, f64)> = items
        .iter()
        .map(|(value, weight)| (value, *weight))
        .collect();
    if data_type == "QUATERNION" && borrowed.iter().all(|(value, _)| is_quaternion(value)) {
        return blend_quaternion(borrowed.iter().copied());
    }
    blend_with_boolean_mixer(borrowed.into_iter(), data_type, weighted_booleans)
}

fn is_quaternion(value: &Value) -> bool {
    value
        .as_array()
        .is_some_and(|array| array.len() == 4 && array.iter().all(Value::is_number))
}

/// Wraps an angle into [-pi, pi), as Blender's `AngleRadianBase::wrapped` (`BLI_math_angle_types.hh:80`).
fn wrap_radians(angle: f32) -> f32 {
    (angle + std::f32::consts::PI).rem_euclid(std::f32::consts::TAU) - std::f32::consts::PI
}

/// Blender's `QuaternionBase::expmap()` (`BLI_math_quaternion.hh:698`) of a native `[x, y, z, w]` quaternion:
/// the axis times the angle `2 * atan2(|xyz|, w)`. A rotation below Blender's `0.0005` sine threshold maps
/// to zero. `None` unless the value is a finite quaternion.
pub(crate) fn quaternion_expmap(value: &Value) -> Option<Vec3> {
    let [x, y, z, w] = value.as_array()?.as_slice() else {
        return None;
    };
    let vector = Vec3::new(x.as_f64()? as f32, y.as_f64()? as f32, z.as_f64()? as f32);
    let real = w.as_f64()? as f32;
    if !vector.is_finite() || !real.is_finite() {
        return None;
    }
    let sin_half = vector.length();
    if sin_half < 0.0005 {
        return Some(Vec3::ZERO);
    }
    let angle = 2.0 * sin_half.atan2(real);
    Some((vector / sin_half) * angle)
}

/// Blender's `QuaternionBase::expmap(VecBase)` (`BLI_math_quaternion.hh:686`): the native `[x, y, z, w]`
/// rotation about the vector's direction by its length, wrapped into [-pi, pi). A zero vector is the identity.
pub(crate) fn quaternion_from_expmap(value: Vec3) -> [f32; 4] {
    let length_squared = value.length_squared();
    if length_squared.is_nan() || length_squared <= 1.0e-35 {
        return [0.0, 0.0, 0.0, 1.0];
    }
    let angle = length_squared.sqrt();
    let half_angle = wrap_radians(angle) * 0.5;
    let normalized = value / angle;
    let imaginary = normalized * half_angle.sin();
    [imaginary.x, imaginary.y, imaginary.z, half_angle.cos()]
}

/// Blender's `DefaultMixer` for `math::Quaternion` (`mix_indices` with `quat_to_expmap` and `expmap_to_quat`,
/// `BKE_attribute_math.hh:431`): the weight-normalised mean of the expmaps, mapped back to a native quaternion.
/// Every quaternion blend uses it. `None` without sources or with a non-finite source.
pub(crate) fn blend_quaternion<'a>(items: impl Iterator<Item = (&'a Value, f64)>) -> Option<Value> {
    let mut value = Vec3::ZERO;
    let mut total_weight = 0.0_f32;
    let mut has_source = false;
    for (source, weight) in items {
        has_source = true;
        value += quaternion_expmap(source)? * (weight as f32);
        total_weight += weight as f32;
    }
    if !has_source {
        return None;
    }
    if total_weight > 0.0 {
        Some(json!(quaternion_from_expmap(
            value * (1.0_f32 / total_weight)
        )))
    } else {
        Some(json!([0.0, 0.0, 0.0, 1.0]))
    }
}

/// Blender's `mix2` of two quaternions (`attribute_math.cc:19`), which is `math::interpolate`: the spherical
/// interpolation along the shorter arc, `a * pow(a^-1 * b, t)`, with `-(a^-1 * b)` when `dot(a, b) < 0`
/// (`BLI_math_quaternion.hh:304`). Values are native `[x, y, z, w]`; `factor` is `t`.
fn quaternion_mix2(first: &Value, second: &Value, factor: f32) -> Option<Value> {
    let a = native_quaternion(first)?;
    let b = native_quaternion(second)?;
    let dot: f32 = a.iter().zip(&b).map(|(x, y)| x * y).sum();
    let mut difference = quaternion_product(quaternion_conjugate(a), b);
    if dot < 0.0 {
        difference = difference.map(|component| -component);
    }
    Some(json!(quaternion_product(
        a,
        quaternion_power(difference, factor)
    )))
}

/// A native `[x, y, z, w]` quaternion as `f32` components, or `None` for a non-finite value.
fn native_quaternion(value: &Value) -> Option<[f32; 4]> {
    let array = value.as_array().filter(|array| array.len() == 4)?;
    let mut components = [0.0_f32; 4];
    for (slot, component) in components.iter_mut().zip(array) {
        let component = component.as_f64()? as f32;
        if !component.is_finite() {
            return None;
        }
        *slot = component;
    }
    Some(components)
}

/// The conjugate of a unit native quaternion, which is its inverse (`invert_normalized`).
fn quaternion_conjugate([x, y, z, w]: [f32; 4]) -> [f32; 4] {
    [-x, -y, -z, w]
}

/// The Hamilton product of native quaternions (`QuaternionBase::operator*`, `BLI_math_quaternion_types.hh:146`).
fn quaternion_product(a: [f32; 4], b: [f32; 4]) -> [f32; 4] {
    let [ax, ay, az, aw] = a;
    let [bx, by, bz, bw] = b;
    [
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by + ay * bw + az * bx - ax * bz,
        aw * bz + az * bw + ax * by - ay * bx,
        aw * bw - ax * bx - ay * by - az * bz,
    ]
}

/// Blender's `pow(q, y)` of a unit quaternion (`BLI_math_quaternion.hh:204`): `cos(y * acos(w))` and
/// `sin(y * acos(w))` times the normalised imaginary part.
fn quaternion_power([x, y, z, w]: [f32; 4], exponent: f32) -> [f32; 4] {
    let half_angle = exponent * w.clamp(-1.0, 1.0).acos();
    let axis = Vec3::new(x, y, z).normalize_or_zero();
    let sine = half_angle.sin();
    [
        axis.x * sine,
        axis.y * sine,
        axis.z * sine,
        half_angle.cos(),
    ]
}

/// The weighted sum of the UVs at `sources`. Indices past the end of `uvs` are skipped, so a corner
/// with no usable source is `[0, 1]`: Blender's zero loop default `(0, 0)` in its V-up space, which
/// is `(0, 1)` in this store's V-down convention (see `blender_uv_loops` in `exchange/blend.rs`).
fn blend_uv(uvs: &[[f64; 2]], sources: &[(usize, f64)]) -> [f64; 2] {
    let mut uv = [0.0; 2];
    let mut found = false;
    for (index, weight) in sources {
        if let Some([u, v]) = uvs.get(*index) {
            found = true;
            uv[0] += weight * u;
            uv[1] += weight * v;
        }
    }
    if found { uv } else { [0.0, 1.0] }
}

/// Blend canonical RGBA colors of `data_type` by source weight, with Blender's `FLOAT_COLOR` and
/// `BYTE_COLOR` mixers. `None` when no source has a weight.
fn blend_rgba(colors: impl Iterator<Item = ([f64; 4], f64)>, data_type: &str) -> Option<[f64; 4]> {
    let byte_color = data_type == "BYTE_COLOR";
    let mut sums = [0.0_f32; 4];
    let mut total = 0.0_f32;
    for (color, weight) in colors {
        let channels = if byte_color {
            color_layers::byte_color_encoded_components(color)
                .map(|channel| channel.round().clamp(0.0, 255.0))
        } else {
            color.map(|channel| channel as f32)
        };
        let weight = weight as f32;
        total += weight;
        for (sum, channel) in sums.iter_mut().zip(channels) {
            *sum += channel * weight;
        }
    }
    if total == 0.0 {
        return None;
    }
    let weight_inv = 1.0_f32 / total;
    Some(if byte_color {
        let bytes = sums.map(|sum| (sum * weight_inv).trunc() as u8 as f32);
        color_layers::byte_color_decoded_components(bytes)
    } else {
        sums.map(|sum| f64::from(sum * weight_inv))
    })
}

/// Blender data types stored as integers; blended results round to the nearest integer.
fn is_integer(data_type: &str) -> bool {
    matches!(data_type, "INT" | "INT8")
}

/// A blended float rounded to the f32 precision Blender stores attribute layers in, so a constant
/// input blends back to exactly its value.
#[expect(
    clippy::cast_possible_truncation,
    reason = "Blender stores float attribute layers as f32"
)]
fn stored_float(sum: f64) -> f64 {
    f64::from(sum as f32)
}

/// The RGBA components of a color value, when it has exactly four numbers.
fn rgba_value(value: &Value) -> Option<[f64; 4]> {
    let components = value
        .as_array()
        .filter(|components| components.len() == 4)?;
    Some([
        components[0].as_f64()?,
        components[1].as_f64()?,
        components[2].as_f64()?,
        components[3].as_f64()?,
    ])
}

/// Blend color values by weight with [`blend_rgba`]. `None` when a value is not RGBA or no source
/// has a weight.
fn blend_color(items: &[(&Value, f64)], data_type: &str) -> Option<Value> {
    let colors = items
        .iter()
        .map(|(value, weight)| Some((rgba_value(value)?, *weight)))
        .collect::<Option<Vec<_>>>()?;
    blend_rgba(colors.into_iter(), data_type)
        .map(|color| Value::Array(color.map(|channel| json!(channel)).into()))
}

/// Blend values by Blender data type: numbers and arrays interpolate componentwise, while
/// `INT16_2D`/`SHORT2` round a float2 accumulation and `FLOAT4X4`/`MATRIX` use Blender's safe TRS
/// mixer. Booleans OR positive-weight sources, and other values use the heaviest source.
/// Boolean corner interpolation uses [`blend_with_boolean_mixer`] for Blender's weighted threshold.
/// `None` when there are no sources.
fn blend<'a>(items: impl Iterator<Item = (&'a Value, f64)>, data_type: &str) -> Option<Value> {
    blend_with_boolean_mixer(items, data_type, false)
}
fn blender_float32(value: f64) -> Option<f32> {
    num_traits::ToPrimitive::to_f32(&value)
}

fn blend_short2(items: &[(&Value, f64)]) -> Option<Value> {
    let mut accum = [0.0_f32; 2];
    let mut total_weight = 0.0_f32;
    for (value, weight) in items {
        let components = value
            .as_array()
            .filter(|components| components.len() == 2)?;
        let weight = blender_float32(*weight)?;
        for (sum, component) in accum.iter_mut().zip(components) {
            let component = i16::try_from(component.as_i64()?).ok()?;
            *sum += f32::from(component) * weight;
        }
        total_weight += weight;
    }
    let result = if total_weight > 0.0 {
        let weight_inv = 1.0_f32 / total_weight;
        [
            num_traits::ToPrimitive::to_i16(&((accum[0] * weight_inv).round()))?,
            num_traits::ToPrimitive::to_i16(&((accum[1] * weight_inv).round()))?,
        ]
    } else {
        [0_i16; 2]
    };
    Some(json!(result))
}

fn float4x4_value(value: &Value) -> Option<[[f32; 4]; 4]> {
    let values = value.as_array()?;
    let mut rows = [[0.0_f32; 4]; 4];
    if values.len() == 4 {
        for (row, source_row) in values.iter().enumerate() {
            let source_row = source_row.as_array().filter(|row| row.len() == 4)?;
            for (column, component) in source_row.iter().enumerate() {
                rows[row][column] = blender_float32(component.as_f64()?)?;
            }
        }
        return Some(rows);
    }
    if values.len() == 16 {
        for row in 0..4 {
            for column in 0..4 {
                rows[row][column] = blender_float32(values[column * 4 + row].as_f64()?)?;
            }
        }
        return Some(rows);
    }
    None
}

/// Blender's `to_loc_rot_scale_safe<true>` for float matrices, using its normalized Euler
/// decomposition before converting the result to a quaternion.
fn float4x4_loc_rot_scale(matrix: [[f32; 4]; 4]) -> ([f32; 3], [f32; 4], [f32; 3]) {
    let location = [matrix[0][3], matrix[1][3], matrix[2][3]];
    let mut rotation_matrix = [[0.0_f32; 3]; 3];
    let mut scale = [0.0_f32; 3];
    for column in 0..3 {
        let x = matrix[0][column];
        let y = matrix[1][column];
        let z = matrix[2][column];
        let length_squared = (x * x + y * y) + z * z;
        if length_squared > 1.0e-35_f32 {
            let length = length_squared.sqrt();
            scale[column] = length;
            rotation_matrix[0][column] = x / length;
            rotation_matrix[1][column] = y / length;
            rotation_matrix[2][column] = z / length;
        }
    }
    let determinant = rotation_matrix[0][0]
        * (rotation_matrix[1][1] * rotation_matrix[2][2]
            - rotation_matrix[1][2] * rotation_matrix[2][1])
        - rotation_matrix[0][1]
            * (rotation_matrix[1][0] * rotation_matrix[2][2]
                - rotation_matrix[1][2] * rotation_matrix[2][0])
        + rotation_matrix[0][2]
            * (rotation_matrix[1][0] * rotation_matrix[2][1]
                - rotation_matrix[1][1] * rotation_matrix[2][0]);
    if determinant < 0.0 {
        for row in &mut rotation_matrix {
            for component in row {
                *component = -*component;
            }
        }
        scale = scale.map(|component| -component);
    }

    let cy = rotation_matrix[0][0].hypot(rotation_matrix[1][0]);
    let (first_euler, second_euler) = if cy > 16.0 * f32::EPSILON {
        (
            [
                rotation_matrix[2][1].atan2(rotation_matrix[2][2]),
                (-rotation_matrix[2][0]).atan2(cy),
                rotation_matrix[1][0].atan2(rotation_matrix[0][0]),
            ],
            [
                (-rotation_matrix[2][1]).atan2(-rotation_matrix[2][2]),
                (-rotation_matrix[2][0]).atan2(-cy),
                (-rotation_matrix[1][0]).atan2(-rotation_matrix[0][0]),
            ],
        )
    } else {
        let euler = [
            (-rotation_matrix[1][2]).atan2(rotation_matrix[1][1]),
            (-rotation_matrix[2][0]).atan2(cy),
            0.0,
        ];
        (euler, euler)
    };
    let first_length = (first_euler[0].abs() + first_euler[1].abs()) + first_euler[2].abs();
    let second_length = (second_euler[0].abs() + second_euler[1].abs()) + second_euler[2].abs();
    let [x, y, z] = if first_length > second_length {
        second_euler
    } else {
        first_euler
    };

    let half_i = x * 0.5;
    let half_j = y * 0.5;
    let half_k = z * 0.5;
    let sin_i = half_i.sin();
    let cos_i = half_i.cos();
    let sin_j = half_j.sin();
    let cos_j = half_j.cos();
    let sin_k = half_k.sin();
    let cos_k = half_k.cos();
    let cos_cos = cos_i * cos_k;
    let cos_sin = cos_i * sin_k;
    let sin_cos = sin_i * cos_k;
    let sin_sin = sin_i * sin_k;
    let quaternion_w = cos_j * cos_cos + sin_j * sin_sin;
    let quaternion_x = cos_j * sin_cos - sin_j * cos_sin;
    let quaternion_y = cos_j * sin_sin + sin_j * cos_cos;
    let quaternion_z = cos_j * cos_sin - sin_j * sin_cos;
    (
        location,
        [quaternion_x, quaternion_y, quaternion_z, quaternion_w],
        scale,
    )
}

fn float4x4_from_loc_rot_scale(
    location: [f32; 3],
    rotation: [f32; 4],
    scale: [f32; 3],
) -> Option<[[f32; 4]; 4]> {
    let [x, y, z, w] = rotation;
    // Blender's float quaternion-to-matrix path uses double intermediates.
    let q0 = std::f64::consts::SQRT_2 * f64::from(w);
    let q1 = std::f64::consts::SQRT_2 * f64::from(x);
    let q2 = std::f64::consts::SQRT_2 * f64::from(y);
    let q3 = std::f64::consts::SQRT_2 * f64::from(z);
    let qda = q0 * q1;
    let qdb = q0 * q2;
    let qdc = q0 * q3;
    let qaa = q1 * q1;
    let qab = q1 * q2;
    let qac = q1 * q3;
    let qbb = q2 * q2;
    let qbc = q2 * q3;
    let qcc = q3 * q3;
    let rotation_matrix = [
        [
            blender_float32(1.0 - qbb - qcc)?,
            blender_float32(-qdc + qab)?,
            blender_float32(qdb + qac)?,
        ],
        [
            blender_float32(qdc + qab)?,
            blender_float32(1.0 - qaa - qcc)?,
            blender_float32(-qda + qbc)?,
        ],
        [
            blender_float32(-qdb + qac)?,
            blender_float32(qda + qbc)?,
            blender_float32(1.0 - qaa - qbb)?,
        ],
    ];
    Some([
        [
            rotation_matrix[0][0] * scale[0],
            rotation_matrix[0][1] * scale[1],
            rotation_matrix[0][2] * scale[2],
            location[0],
        ],
        [
            rotation_matrix[1][0] * scale[0],
            rotation_matrix[1][1] * scale[1],
            rotation_matrix[1][2] * scale[2],
            location[1],
        ],
        [
            rotation_matrix[2][0] * scale[0],
            rotation_matrix[2][1] * scale[1],
            rotation_matrix[2][2] * scale[2],
            location[2],
        ],
        [0.0, 0.0, 0.0, 1.0],
    ])
}

fn float4x4_quaternion_expmap([x, y, z, w]: [f32; 4]) -> [f32; 3] {
    let sin_half = ((x * x + y * y) + z * z).sqrt();
    if sin_half < 0.0005 {
        return [0.0; 3];
    }
    let angle = 2.0 * sin_half.atan2(w);
    [
        (x / sin_half) * angle,
        (y / sin_half) * angle,
        (z / sin_half) * angle,
    ]
}

fn float4x4_rotation_from_expmap([x, y, z]: [f32; 3]) -> [f32; 4] {
    let length_squared = (x * x + y * y) + z * z;
    if length_squared > 1.0e-35_f32 {
        let angle = length_squared.sqrt();
        let wrapped_angle = angle + std::f32::consts::PI;
        let wrapped_angle =
            wrapped_angle - std::f32::consts::TAU * (wrapped_angle / std::f32::consts::TAU).floor();
        let half_angle = (wrapped_angle - std::f32::consts::PI) * 0.5;
        let sine = half_angle.sin();
        [
            (x / angle) * sine,
            (y / angle) * sine,
            (z / angle) * sine,
            half_angle.cos(),
        ]
    } else {
        [0.0, 0.0, 0.0, 1.0]
    }
}

fn blend_float4x4(items: &[(&Value, f64)]) -> Option<Value> {
    let mut location = [0.0_f32; 3];
    let mut expmap = [0.0_f32; 3];
    let mut scale = [0.0_f32; 3];
    let mut total_weight = 0.0_f32;
    for (value, weight) in items {
        let matrix = float4x4_value(value)?;
        let (source_location, source_rotation, source_scale) = float4x4_loc_rot_scale(matrix);
        let source_expmap = float4x4_quaternion_expmap(source_rotation);
        let weight = blender_float32(*weight)?;
        for component in 0..3 {
            location[component] += source_location[component] * weight;
            expmap[component] += source_expmap[component] * weight;
            scale[component] += source_scale[component] * weight;
        }
        total_weight += weight;
    }
    let result = if total_weight > 0.0 {
        let weight_inv = 1.0_f32 / total_weight;
        let rotation = float4x4_rotation_from_expmap(expmap.map(|value| value * weight_inv));
        float4x4_from_loc_rot_scale(
            location.map(|value| value * weight_inv),
            rotation,
            scale.map(|value| value * weight_inv),
        )?
    } else {
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    };
    Some(json!(result))
}

fn is_short2(data_type: &str) -> bool {
    matches!(data_type, "INT16_2D" | "SHORT2")
}

fn is_float4x4(data_type: &str) -> bool {
    matches!(data_type, "FLOAT4X4" | "MATRIX")
}

fn blend_with_boolean_mixer<'a>(
    items: impl Iterator<Item = (&'a Value, f64)>,
    data_type: &str,
    weighted_booleans: bool,
) -> Option<Value> {
    let items: Vec<(&Value, f64)> = items.collect();
    let first = items.first()?.0;
    if matches!(data_type, "FLOAT_COLOR" | "BYTE_COLOR")
        && let Some(color) = blend_color(&items, data_type)
    {
        return Some(color);
    }
    if is_short2(data_type) {
        return blend_short2(&items);
    }
    if is_float4x4(data_type) {
        return blend_float4x4(&items);
    }
    if items.iter().all(|(value, _)| value.is_number()) {
        let sum: f64 = items
            .iter()
            .filter_map(|(value, weight)| Some(value.as_f64()? * weight))
            .sum();
        return Some(if is_integer(data_type) {
            json!(sum.round() as i64)
        } else {
            json!(stored_float(sum))
        });
    }
    if data_type == "QUATERNION" && items.iter().all(|(value, _)| is_quaternion(value)) {
        // A lone full-weight source is copied, as Blender copies a coarse element without mixing it.
        if let [(value, weight)] = items.as_slice()
            && *weight > 1.0 - 1.0e-9
        {
            return Some((*value).clone());
        }
        return blend_quaternion(items.iter().copied());
    }
    if let Some(length) = first.as_array().map(Vec::len)
        && items.iter().all(|(value, _)| {
            value
                .as_array()
                .is_some_and(|array| array.len() == length && array.iter().all(Value::is_number))
        })
    {
        let mut sums = vec![0.0; length];
        for (value, weight) in &items {
            for (sum, component) in sums.iter_mut().zip(value.as_array().into_iter().flatten()) {
                *sum += weight * component.as_f64().unwrap_or_default();
            }
        }
        let rounds = data_type == "INT32_2D";
        return Some(Value::Array(
            sums.into_iter()
                .map(|sum| {
                    if rounds {
                        json!(sum.round() as i64)
                    } else {
                        json!(stored_float(sum))
                    }
                })
                .collect(),
        ));
    }
    if items.iter().all(|(value, _)| value.is_boolean()) {
        if !weighted_booleans {
            return Some(json!(items.iter().any(|(value, weight)| {
                *weight > 0.0 && value.as_bool() == Some(true)
            })));
        }
        let mut sum = 0.0_f32;
        let mut total = 0.0_f32;
        for (value, weight) in &items {
            sum += (if value.as_bool() == Some(true) {
                1.0
            } else {
                0.0
            }) * (*weight as f32);
            total += *weight as f32;
        }
        return Some(json!(total > 0.0 && sum * (1.0 / total) >= 0.5));
    }
    items
        .iter()
        .max_by(|(_, left), (_, right)| left.total_cmp(right))
        .map(|(value, _)| (*value).clone())
}

/// Value written for an element with no source, by Blender data type.
pub(crate) fn default_value(data_type: &str) -> Value {
    match data_type {
        "BOOLEAN" | "BOOL" => json!(false),
        "INT" | "INT8" => json!(0),
        "FLOAT" => json!(0.0),
        "INT32_2D" | "INT16_2D" | "SHORT2" => json!([0, 0]),
        "FLOAT4X4" | "MATRIX" => json!([
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ]),
        "FLOAT2" => json!([0.0, 0.0]),
        "FLOAT_VECTOR" => json!([0.0, 0.0, 0.0]),
        "FLOAT_COLOR" | "BYTE_COLOR" | "QUATERNION" => json!([0.0, 0.0, 0.0, 1.0]),
        _ => Value::Null,
    }
}

/// The Blender data type that [`blend`] and [`default_value`] use for a native attribute type.
fn blend_type(kind: &str) -> &'static str {
    match kind {
        "int" => "INT",
        "bool" => "BOOLEAN",
        "float2" => "FLOAT2",
        "float3" => "FLOAT_VECTOR",
        "quaternion" => "QUATERNION",
        "color" => "FLOAT_COLOR",
        _ => "FLOAT",
    }
}

/// [`blend`] over the `stored` values at the `(index, weight)` sources. Sources without a value are skipped.
fn blend_sources(
    stored: &[Option<&Value>],
    sources: &[(usize, f64)],
    data_type: &str,
) -> Option<Value> {
    blend_mix(
        |index| stored.get(index).copied().flatten(),
        Mix::Flat(sources),
        data_type,
        false,
    )
}

/// Whether remapping `source` under `skipped` reads per-corner source weights: a kept corner-domain generic
/// attribute, a kept corner-domain colour layer, or kept corner-domain UV, custom-normal, or legacy corner
/// data. Otherwise the remap never reads corner sources, so the caller may leave them empty.
pub(crate) fn needs_corner_sources(source: &Mesh, skipped: &[&str]) -> bool {
    let kept = |key: &str| !skipped.contains(&key);
    let has = |key: &str| kept(key) && source.attributes.contains_key(key);
    let entries = source
        .attributes
        .get("blender_attributes")
        .and_then(Value::as_array)
        .filter(|_| kept("blender_attributes"))
        .map_or(&[][..], Vec::as_slice);
    has(uv_layers::ATTRIBUTE)
        || has("blender_uv_layers")
        || has("uv_map")
        || has("custom_normal")
        || entries
            .iter()
            .any(|entry| text(entry, "domain") == "CORNER")
        || source
            .attributes
            .iter()
            .filter(|(key, _)| kept(key.as_str()))
            .any(|(_, value)| named_domain(value).as_deref() == Some("corner"))
        || (has(color_layers::ATTRIBUTE)
            && color_layers::read(source)
                .layers
                .iter()
                .any(|layer| matches!(layer.domain, ColorDomain::Corner)))
}

/// The element domain of a native attribute (`point`, `edge`, `face`, or `corner`). Other mesh keys, such
/// as `uv_layers`, `color_layers`, `vertex_groups`, and the Blender containers, have no such domain.
fn native_domain(value: &Value) -> Option<&str> {
    value
        .get("domain")
        .and_then(Value::as_str)
        .filter(|domain| matches!(*domain, "point" | "edge" | "face" | "corner"))
}

/// Flags of the edge-domain boolean attribute `name`, one per edge in storage order. An edge without an
/// entry, or a mesh without the attribute, is unflagged.
pub(crate) fn edge_flags(mesh: &Mesh, name: &str) -> Vec<bool> {
    let values = mesh
        .attributes
        .get(name)
        .and_then(|attribute| attribute.get("values"))
        .and_then(Value::as_object);
    mesh.edges
        .iter()
        .map(|edge| {
            values
                .and_then(|values| values.get(&format!("e{}", edge.id)))
                .and_then(Value::as_bool)
                == Some(true)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    #![expect(clippy::expect_used, reason = "small mesh fixtures")]

    use glam::DVec3;
    use serde_json::json;

    use super::color_layers::ColorStore;
    use super::*;

    fn triangle() -> Mesh {
        Mesh::from_positions_and_faces(
            vec![
                DVec3::new(0.0, 0.0, 0.0),
                DVec3::new(1.0, 0.0, 0.0),
                DVec3::new(0.0, 1.0, 0.0),
            ],
            vec![vec![0, 1, 2]],
        )
        .expect("triangle")
    }

    fn expected_byte_color(bytes: [f32; 4]) -> [f64; 4] {
        const BYTE_TO_UNIT: f32 = 1.0 / 255.0;
        std::array::from_fn(|index| {
            let encoded = bytes[index] * BYTE_TO_UNIT;
            let decoded = if index < 3 {
                if encoded <= 0.04045 {
                    encoded * (1.0 / 12.92)
                } else {
                    ((encoded + 0.055) * (1.0 / 1.055)).powf(2.4)
                }
            } else {
                encoded
            };
            f64::from(decoded)
        })
    }

    #[test]
    fn interpolated_vertex_group_blends_source_weights() {
        let mut source = triangle();
        let ids: Vec<u32> = source.vertices.iter().map(|vertex| vertex.id).collect();
        source.attributes.insert(
            "vertex_groups".to_owned(),
            json!({"Arm": {format!("v{}", ids[0]): 1.0, format!("v{}", ids[1]): 0.0}}),
        );
        let mut output = triangle();
        let origin = Origin {
            vertices: vec![vec![(0, 1.0)], vec![(1, 1.0)], vec![(0, 0.5), (1, 0.5)]],
            source_vertices: None,
            faces: vec![Some(0)],
            corners: None,
            ..Default::default()
        };
        remap_attributes(&source, &mut output, &origin).expect("remap");
        let out_ids: Vec<u32> = output.vertices.iter().map(|vertex| vertex.id).collect();
        let groups = &output.attributes["vertex_groups"]["Arm"];
        assert_eq!(groups[format!("v{}", out_ids[0])], json!(1.0));
        assert_eq!(groups[format!("v{}", out_ids[1])], json!(0.0));
        assert_eq!(groups[format!("v{}", out_ids[2])], json!(0.5));
    }

    #[test]
    fn origindex_carrier_follows_single_weight_one_sources() {
        let mut source = triangle();
        crate::geom::set_origindex(&mut source, &[1, ORIGINDEX_NONE, 2]);

        let mut output = triangle();
        let copied = Origin::copies([Some(2), None, Some(0)], vec![Some(0)]);
        remap_attributes(&source, &mut output, &copied).expect("remap copied indices");
        assert_eq!(
            crate::geom::origindex_values(&output),
            Some(vec![2, ORIGINDEX_NONE, 1])
        );

        let mut output = triangle();
        let blended = Origin {
            vertices: vec![vec![(0, 1.0)], vec![(1, 1.0)], vec![(0, 0.5), (1, 0.5)]],
            source_vertices: None,
            faces: vec![Some(0)],
            corners: None,
            ..Default::default()
        };
        remap_attributes(&source, &mut output, &blended).expect("remap blended indices");
        assert_eq!(
            crate::geom::origindex_values(&output),
            Some(vec![1, ORIGINDEX_NONE, ORIGINDEX_NONE])
        );
    }

    /// A JSON object of keyed entries, for attribute fixtures.
    fn keyed(entries: Vec<(String, Value)>) -> Value {
        Value::Object(entries.into_iter().collect())
    }

    /// Subdivision of one quad: edge midpoints take two corners, the face centre takes all four.
    /// Boundary halves keep their source edge's sharp and seam flags; the inner spokes are unflagged.
    #[test]
    fn subdivision_inner_edges_take_no_source_edge_flags() {
        let corners = vec![
            DVec3::new(0.0, 0.0, 0.0),
            DVec3::new(1.0, 0.0, 0.0),
            DVec3::new(1.0, 1.0, 0.0),
            DVec3::new(0.0, 1.0, 0.0),
        ];
        let mut source =
            Mesh::from_positions_and_faces(corners, vec![vec![0, 1, 2, 3]]).expect("quad");
        let flags = keyed(
            source
                .edges
                .iter()
                .map(|edge| (format!("e{}", edge.id), json!(true)))
                .collect(),
        );
        source.attributes.insert(
            "sharp".to_owned(),
            json!({"domain": "edge", "type": "bool", "values": flags.clone()}),
        );
        source.attributes.insert(
            "seam".to_owned(),
            json!({"domain": "edge", "type": "bool", "values": flags}),
        );
        let positions = vec![
            DVec3::new(0.0, 0.0, 0.0),
            DVec3::new(1.0, 0.0, 0.0),
            DVec3::new(1.0, 1.0, 0.0),
            DVec3::new(0.0, 1.0, 0.0),
            DVec3::new(0.5, 0.0, 0.0),
            DVec3::new(1.0, 0.5, 0.0),
            DVec3::new(0.5, 1.0, 0.0),
            DVec3::new(0.0, 0.5, 0.0),
            DVec3::new(0.5, 0.5, 0.0),
        ];
        let mut output = Mesh::from_positions_and_faces(
            positions,
            vec![
                vec![0, 4, 8, 7],
                vec![4, 1, 5, 8],
                vec![8, 5, 2, 6],
                vec![7, 8, 6, 3],
            ],
        )
        .expect("subdivided quad");
        let origin = Origin {
            vertices: vec![
                vec![(0, 1.0)],
                vec![(1, 1.0)],
                vec![(2, 1.0)],
                vec![(3, 1.0)],
                vec![(0, 0.5), (1, 0.5)],
                vec![(1, 0.5), (2, 0.5)],
                vec![(2, 0.5), (3, 0.5)],
                vec![(3, 0.5), (0, 0.5)],
                vec![(0, 0.25), (1, 0.25), (2, 0.25), (3, 0.25)],
            ],
            source_vertices: None,
            faces: vec![Some(0); 4],
            corners: None,
            ..Default::default()
        };
        remap_attributes(&source, &mut output, &origin).expect("remap");
        let centre = output.vertices[8].id;
        assert_eq!(output.edges.len(), 12);
        let sharp = edge_flags(&output, "sharp");
        let seam = edge_flags(&output, "seam");
        for (edge, (sharp, seam)) in output.edges.iter().zip(sharp.into_iter().zip(seam)) {
            let boundary = !edge.vertices.contains(&centre);
            assert_eq!((sharp, seam), (boundary, boundary));
        }
    }

    /// Weld merges three coincident edges/vertices. Group propagation is unweighted OR for seam,
    /// sharp, and generic POINT/EDGE booleans, even when only one of three sources is true.
    #[test]
    fn welded_boolean_group_attributes_keep_unweighted_or() {
        let mut source = Mesh::new();
        let vertices = [
            source.insert_vertex(DVec3::ZERO).expect("vertex"),
            source.insert_vertex(DVec3::X).expect("vertex"),
            source.insert_vertex(DVec3::ZERO).expect("vertex"),
            source.insert_vertex(DVec3::X).expect("vertex"),
            source.insert_vertex(DVec3::ZERO).expect("vertex"),
            source.insert_vertex(DVec3::X).expect("vertex"),
        ];
        for pair in vertices.as_chunks::<2>().0 {
            source.insert_edge([pair[0], pair[1]]).expect("source edge");
        }
        let edge_ids: Vec<u32> = source.edges.iter().map(|edge| edge.id).collect();
        let vertex_values = keyed(vec![
            (format!("v{}", vertices[0]), json!(true)),
            (format!("v{}", vertices[1]), json!(false)),
            (format!("v{}", vertices[2]), json!(false)),
            (format!("v{}", vertices[3]), json!(false)),
            (format!("v{}", vertices[4]), json!(false)),
            (format!("v{}", vertices[5]), json!(false)),
        ]);
        let edge_values = keyed(vec![
            (format!("e{}", edge_ids[0]), json!(true)),
            (format!("e{}", edge_ids[1]), json!(false)),
            (format!("e{}", edge_ids[2]), json!(false)),
        ]);
        for name in ["sharp", "seam", "edge_mask"] {
            source.attributes.insert(
                name.to_owned(),
                json!({"domain":"edge", "type":"bool", "values":edge_values.clone()}),
            );
        }
        source.attributes.insert(
            "point_mask".to_owned(),
            json!({"domain":"point", "type":"bool", "values":vertex_values}),
        );
        let mut output = Mesh::new();
        let near = output.insert_vertex(DVec3::ZERO).expect("vertex");
        let far = output.insert_vertex(DVec3::X).expect("vertex");
        output.insert_edge([near, far]).expect("output edge");
        let origin = Origin {
            vertices: vec![
                vec![(0, 1.0 / 3.0), (2, 1.0 / 3.0), (4, 1.0 / 3.0)],
                vec![(1, 1.0 / 3.0), (3, 1.0 / 3.0), (5, 1.0 / 3.0)],
            ],
            source_vertices: None,
            faces: Vec::new(),
            corners: None,
            ..Default::default()
        };
        remap_attributes(&source, &mut output, &origin).expect("Weld attribute remap");
        assert_eq!(edge_flags(&output, "sharp"), vec![true]);
        assert_eq!(edge_flags(&output, "seam"), vec![true]);
        assert_eq!(
            output.attributes["edge_mask"]["values"][format!("e{}", output.edges[0].id)],
            json!(true)
        );
        assert_eq!(
            output.attributes["point_mask"]["values"][format!("v{}", output.vertices[0].id)],
            json!(true)
        );
    }

    /// Native element attributes follow their origins: a point color blends the two sources of a merged
    /// vertex, a corner color follows the corner it came from (blended where the output corner has two
    /// sources), and a face value copies its source face.
    #[test]
    fn native_color_and_generic_attributes_follow_origins() {
        let mut source = triangle();
        let vertices: Vec<u32> = source.vertices.iter().map(|vertex| vertex.id).collect();
        let face = source.faces[0].id;
        source.attributes.insert(
            "Col".to_owned(),
            json!({"domain": "point", "type": "color", "values": keyed(vec![
                (format!("v{}", vertices[0]), json!([1.0, 0.0, 0.0, 1.0])),
                (format!("v{}", vertices[1]), json!([0.0, 1.0, 0.0, 1.0])),
                (format!("v{}", vertices[2]), json!([0.0, 0.0, 1.0, 1.0])),
            ])}),
        );
        source.attributes.insert(
            "Paint".to_owned(),
            json!({"domain": "corner", "type": "color", "values": keyed(vec![(
                format!("f{face}"),
                json!([[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0]]),
            )])}),
        );
        source.attributes.insert(
            "weight".to_owned(),
            json!({"domain": "face", "type": "float", "values": keyed(vec![(
                format!("f{face}"),
                json!(0.75),
            )])}),
        );
        let mut output = triangle();
        let origin = Origin {
            vertices: vec![vec![(0, 1.0)], vec![(1, 1.0)], vec![(0, 0.5), (1, 0.5)]],
            source_vertices: None,
            faces: vec![Some(0)],
            corners: None,
            ..Default::default()
        };
        remap_attributes(&source, &mut output, &origin).expect("remap");
        let out_vertex = format!("v{}", output.vertices[2].id);
        let out_face = format!("f{}", output.faces[0].id);
        assert_eq!(
            output.attributes["Col"]["values"][out_vertex.as_str()],
            json!([0.5, 0.5, 0.0, 1.0])
        );
        assert_eq!(
            output.attributes["Paint"]["values"][out_face.as_str()],
            json!([
                [1.0, 0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0, 1.0],
                [0.5, 0.5, 0.0, 1.0]
            ])
        );
        assert_eq!(
            output.attributes["weight"]["values"][out_face.as_str()],
            json!(0.75)
        );
    }

    #[test]
    fn canonical_color_store_layers_follow_origins_and_keep_designations() {
        let mut source = triangle();
        let red = [1.0, 0.0, 0.0, 1.0];
        let green = [0.0, 1.0, 0.0, 1.0];
        let blue = [0.0, 0.0, 1.0, 1.0];
        let store = color_layers::ColorStore {
            layers: vec![
                ColorLayer {
                    name: "color_layers".to_owned(),
                    domain: ColorDomain::Point,
                    values: vec![red, green, blue],
                },
                ColorLayer {
                    name: "Paint".to_owned(),
                    domain: ColorDomain::Corner,
                    values: vec![red, green, blue],
                },
            ],
            active: Some("Paint".to_owned()),
            render: Some("color_layers".to_owned()),
        };
        color_layers::write(&mut source, &store);
        assert!(
            source
                .attributes
                .contains_key(attribute_keys::storage_key("color_layers").as_ref())
        );

        let mut output = triangle();
        let origin = Origin {
            vertices: vec![vec![(0, 1.0)], vec![(1, 1.0)], vec![(0, 0.5), (1, 0.5)]],
            source_vertices: None,
            faces: vec![Some(0)],
            corners: None,
            ..Default::default()
        };
        remap_attributes(&source, &mut output, &origin).expect("remap");

        let store = color_layers::read(&output);
        assert_eq!(
            store
                .layers
                .iter()
                .map(|layer| layer.name.as_str())
                .collect::<Vec<_>>(),
            vec!["color_layers", "Paint"]
        );
        assert_eq!(store.layers[0].domain, ColorDomain::Point);
        assert_eq!(
            store.layers[0].values,
            vec![red, green, [0.5, 0.5, 0.0, 1.0]]
        );
        assert_eq!(store.layers[1].domain, ColorDomain::Corner);
        assert_eq!(
            store.layers[1].values,
            vec![red, green, [0.5, 0.5, 0.0, 1.0]]
        );
        assert_eq!(store.active.as_deref(), Some("Paint"));
        assert_eq!(store.render.as_deref(), Some("color_layers"));
    }

    #[test]
    fn color_mixers_preserve_float32_and_byte_color_quantization() {
        let black = json!([0.0, 0.0, 0.0, 0.0]);
        let white = json!([1.0, 1.0, 1.0, 1.0]);
        assert_eq!(
            blend([(&black, 0.5), (&white, 0.5)].into_iter(), "FLOAT_COLOR"),
            Some(json!([0.5, 0.5, 0.5, 0.5]))
        );
        assert_eq!(
            blend([(&black, 1.0), (&white, 3.0)].into_iter(), "FLOAT_COLOR"),
            Some(json!([0.75, 0.75, 0.75, 0.75]))
        );
        let large = json!([16_777_216.0, 0.0, 0.0, 1.0]);
        let small = json!([1.0, 0.0, 0.0, 0.0]);
        assert_eq!(
            blend([(&large, 0.5), (&small, 0.5)].into_iter(), "FLOAT_COLOR"),
            Some(json!([8_388_608.0, 0.0, 0.0, 0.5]))
        );
        let byte_half = blend([(&black, 0.5), (&white, 0.5)].into_iter(), "BYTE_COLOR")
            .expect("byte color blend");
        assert_eq!(byte_half, json!(expected_byte_color([127.0; 4])));

        let halfway = json!([0.5, 0.5, 0.5, 0.5]);
        let rounded =
            blend([(&halfway, 1.0)].into_iter(), "BYTE_COLOR").expect("single byte color");
        assert_eq!(
            rounded,
            json!(expected_byte_color([188.0, 188.0, 188.0, 128.0]))
        );
    }

    #[test]
    fn color_encoding_comes_from_native_metadata_and_generic_data_type() {
        let mut source = triangle();
        let vertices: Vec<u32> = source.vertices.iter().map(|vertex| vertex.id).collect();
        let black = json!([0.0, 0.0, 0.0, 0.0]);
        let white = json!([1.0, 1.0, 1.0, 1.0]);
        let values = vec![black.clone(), white.clone(), white.clone()];
        let native_values = keyed(vec![
            (format!("v{}", vertices[0]), black.clone()),
            (format!("v{}", vertices[1]), white.clone()),
            (format!("v{}", vertices[2]), white.clone()),
        ]);
        let escaped_name = attribute_keys::storage_key("color_layers").into_owned();
        source.attributes.insert(
            escaped_name.clone(),
            json!({"domain": "point", "type": "color", "values": native_values.clone()}),
        );
        source.attributes.insert(
            "Fallback".to_owned(),
            json!({"domain": "point", "type": "color", "values": native_values.clone()}),
        );
        source.attributes.insert(
            "FloatByMetadata".to_owned(),
            json!({"domain": "point", "type": "color", "values": native_values}),
        );
        source.attributes.insert(
            "blender_metadata".to_owned(),
            json!({"color_data_types": {
                "color_layers": "BYTE_COLOR",
                "FloatByMetadata": "FLOAT_COLOR"
            }}),
        );
        source.attributes.insert(
            "blender_attributes".to_owned(),
            json!([
                {"name": "Byte", "domain": "POINT", "data_type": "BYTE_COLOR", "values": values},
                {
                    "name": "Float",
                    "domain": "POINT",
                    "data_type": "FLOAT_COLOR",
                    "values": [black.clone(), white.clone(), white.clone()]
                }
            ]),
        );
        let mut output = triangle();
        let origin = Origin {
            vertices: vec![vec![(0, 1.0)], vec![(1, 1.0)], vec![(0, 0.5), (1, 0.5)]],
            source_vertices: None,
            faces: vec![Some(0)],
            corners: None,
            ..Default::default()
        };

        remap_attributes(&source, &mut output, &origin).expect("remap");

        let output_vertex = format!("v{}", output.vertices[2].id);
        let expected_byte = json!(expected_byte_color([127.0; 4]));
        let expected_float = json!([0.5, 0.5, 0.5, 0.5]);
        assert_eq!(
            output.attributes[&escaped_name]["values"][output_vertex.as_str()],
            expected_byte
        );
        assert_eq!(
            output.attributes["Fallback"]["values"][output_vertex.as_str()],
            expected_float
        );
        assert_eq!(
            output.attributes["FloatByMetadata"]["values"][output_vertex.as_str()],
            expected_float
        );
        assert_eq!(
            output.attributes["blender_attributes"][0]["values"][2],
            expected_byte
        );
        assert_eq!(
            output.attributes["blender_attributes"][1]["values"][2],
            expected_float
        );
    }

    #[test]
    fn byte_color_corner_remap_uses_encoded_bytes_float_color_remains_linear() {
        let mut source = triangle();
        let black = [0.0, 0.0, 0.0, 1.0];
        let white = [1.0, 1.0, 1.0, 1.0];
        color_layers::write(
            &mut source,
            &ColorStore {
                layers: vec![
                    ColorLayer {
                        name: "Byte".to_owned(),
                        domain: ColorDomain::Corner,
                        values: vec![black, white, black],
                    },
                    ColorLayer {
                        name: "Float".to_owned(),
                        domain: ColorDomain::Corner,
                        values: vec![black, white, black],
                    },
                ],
                active: Some("Float".to_owned()),
                render: Some("Float".to_owned()),
            },
        );
        source.attributes.insert(
            "blender_metadata".to_owned(),
            json!({"color_data_types":{"Byte":"BYTE_COLOR","Float":"FLOAT_COLOR"}}),
        );
        let mut output = triangle();
        let origin = Origin {
            vertices: vec![vec![(0, 1.0)], vec![(1, 1.0)], vec![(2, 1.0)]],
            source_vertices: None,
            faces: vec![Some(0)],
            corners: Some(vec![
                vec![(0, 0.49), (1, 0.51)],
                vec![(1, 1.0)],
                vec![(2, 1.0)],
            ]),
            ..Default::default()
        };
        remap_attributes(&source, &mut output, &origin).expect("remap colors");
        let result = color_layers::read(&output);
        let byte = result
            .layers
            .iter()
            .find(|layer| layer.name == "Byte")
            .expect("byte layer");
        let float = result
            .layers
            .iter()
            .find(|layer| layer.name == "Float")
            .expect("float layer");
        let byte_channel = expected_byte_color([130.0; 4])[0];
        assert!((byte.values[0][0] - byte_channel).abs() < 1.0e-12);
        assert!((float.values[0][0] - 0.51).abs() < 1.0e-6);
    }

    #[test]
    fn group_boolean_mixer_ors_while_weighted_corner_mixer_uses_threshold() {
        let on = json!(true);
        let off = json!(false);
        assert_eq!(
            blend([(&on, 1.0 / 3.0), (&off, 2.0 / 3.0)].into_iter(), "BOOLEAN"),
            Some(json!(true))
        );
        assert_eq!(
            blend([(&on, 0.0), (&off, 1.0)].into_iter(), "BOOLEAN"),
            Some(json!(false))
        );
        assert_eq!(
            blend_with_boolean_mixer([(&on, 0.49), (&off, 0.51)].into_iter(), "BOOLEAN", true,),
            Some(json!(false))
        );
        assert_eq!(
            blend_with_boolean_mixer([(&on, 0.5), (&off, 0.5)].into_iter(), "BOOLEAN", true,),
            Some(json!(true))
        );
    }

    #[test]
    fn short2_mixer_rounds_weighted_f32_values_and_uses_zero_default() {
        let first = json!([1, -1]);
        let second = json!([2, 0]);
        assert_eq!(
            blend_with_boolean_mixer(
                [(&first, 0.5), (&second, 0.5)].into_iter(),
                "INT16_2D",
                false,
            ),
            Some(json!([2, -1]))
        );
        assert_eq!(
            blend_with_boolean_mixer([(&first, 0.0)].into_iter(), "SHORT2", false),
            Some(json!([0, 0]))
        );
        assert_eq!(default_value("SHORT2"), json!([0, 0]));
    }

    #[test]
    fn float4x4_mixer_reads_rows_and_column_major_values() {
        let identity = json!([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]);
        let translated = json!([
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 2.0, 0.0, 0.0, 1.0,
        ]);
        let expected = json!([
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]);
        assert_eq!(
            blend_with_boolean_mixer(
                [(&identity, 0.5), (&translated, 0.5)].into_iter(),
                "FLOAT4X4",
                false,
            ),
            Some(expected)
        );
        assert_eq!(
            blend_with_boolean_mixer([(&translated, 0.0)].into_iter(), "MATRIX", false),
            Some(json!([
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]))
        );
        assert_eq!(
            default_value("FLOAT4X4"),
            json!([
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ])
        );
    }

    #[test]
    fn float4x4_mixer_blends_rotation_in_expmap_space() {
        let identity = json!([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]);
        let quarter_turn = json!([
            [0.0, -1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]);
        let mixed = blend_with_boolean_mixer(
            [(&identity, 0.5), (&quarter_turn, 0.5)].into_iter(),
            "MATRIX",
            false,
        )
        .expect("matrix mix");
        assert!(mixed[0][1].as_f64().is_some_and(|value| value < 0.0));
        assert!(mixed[1][0].as_f64().is_some_and(|value| value > 0.0));
    }

    #[test]
    fn interpolated_corner_booleans_use_weighted_mixer_without_changing_group_merge_or() {
        let mut source = triangle();
        let face = source.faces[0].id;
        source.attributes.insert(
            "corner_mask".to_owned(),
            json!({"domain":"corner", "type":"bool", "values":keyed(vec![(
                format!("f{face}"), json!([true, false, false]),
            )])}),
        );
        source.attributes.insert(
            "blender_attributes".to_owned(),
            json!([{"name":"corner mask", "domain":"CORNER", "data_type":"BOOLEAN", "values":[true, false, false]}]),
        );
        source.attributes.insert(
            "named_corner_mask".to_owned(),
            json!({"domain":"CORNER", "type":"bool", "values":keyed(vec![(
                format!("f{face}"), json!([true, false, false]),
            )])}),
        );
        let origin = Origin {
            vertices: vec![vec![(0, 1.0)], vec![(1, 1.0)], vec![(2, 1.0)]],
            source_vertices: None,
            faces: vec![Some(0)],
            corners: Some(vec![
                vec![(0, 0.49), (1, 0.51)],
                vec![(1, 1.0)],
                vec![(2, 1.0)],
            ]),
            ..Default::default()
        };
        let face_key = format!("f{face}");
        let mut boolean_output = triangle();
        remap_interpolated_corners_skipping(&source, &mut boolean_output, &origin, &[])
            .expect("Boolean attribute remap");
        assert_eq!(
            boolean_output.attributes["corner_mask"]["values"][face_key.as_str()][0],
            json!(false)
        );
        assert_eq!(
            boolean_output.attributes["blender_attributes"][0]["values"][0],
            json!(false)
        );
        assert_eq!(
            boolean_output.attributes["named_corner_mask"]["values"][face_key.as_str()][0],
            json!(false)
        );

        let mut group_output = triangle();
        remap_attributes(&source, &mut group_output, &origin).expect("group attribute remap");
        assert_eq!(
            group_output.attributes["corner_mask"]["values"][face_key.as_str()][0],
            json!(true)
        );
        assert_eq!(
            group_output.attributes["blender_attributes"][0]["values"][0],
            json!(true)
        );
        assert_eq!(
            group_output.attributes["named_corner_mask"]["values"][face_key.as_str()][0],
            json!(true)
        );
    }

    #[test]
    fn mean_value_corners_use_blenders_scale_aware_edge_snap() {
        let scale = 1_000_000.0;
        let source = Mesh::from_positions_and_faces(
            vec![
                DVec3::ZERO,
                DVec3::new(scale, 0.0, 0.0),
                DVec3::new(scale, scale, 0.0),
                DVec3::new(0.0, scale, 0.0),
            ],
            vec![vec![0, 1, 2, 3]],
        )
        .expect("source quad");
        let output_corners = [0];
        let weights_at =
            |point| mean_value_corners(&source, &[point], [(output_corners.as_slice(), Some(0))]);
        assert_eq!(
            weights_at(DVec3::new(scale * 0.5, 0.0, 0.0)),
            vec![vec![(0, 0.5), (1, 0.5)]],
        );
        assert_eq!(weights_at(DVec3::ZERO), vec![vec![(0, 1.0)]]);
        assert_eq!(
            weights_at(DVec3::new(scale * 0.5, 0.25, 0.0)),
            vec![vec![(0, 0.5), (1, 0.5)]],
        );
    }

    /// A rebuilt corner keeps the pin of its heaviest source corner (ties: lowest source index),
    /// and `pinned` stays aligned with the output corners.
    #[test]
    fn uv_pins_follow_heaviest_source_corner() {
        let mut source = triangle();
        uv_layers::write_layers(
            &mut source,
            &[UvLayer {
                name: "UVMap".to_owned(),
                faces: vec![Some(vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])],
                pinned: vec![Some(vec![true, false, false])],
            }],
        );
        let mut output = triangle();
        let origin = Origin {
            vertices: vec![vec![(0, 1.0)], vec![(1, 1.0)], vec![(2, 1.0)]],
            source_vertices: None,
            faces: vec![Some(0)],
            corners: Some(vec![
                vec![(0, 0.25), (1, 0.75)],
                vec![(0, 0.5), (1, 0.5)],
                vec![(2, 1.0)],
            ]),
            ..Default::default()
        };
        remap_attributes(&source, &mut output, &origin).expect("remap");
        let layers = uv_layers::read_layers(&output);
        assert_eq!(layers[0].faces[0].as_ref().map(Vec::len), Some(3));
        assert_eq!(layers[0].pinned[0], Some(vec![false, true, false]));
    }

    /// Screw can create a face with no source face while carrying corner provenance from a partial
    /// UV layer; missing and source-less corners take `[0, 1]`, Blender's zero loop default
    /// `(0, 0)` flipped into this store's V-down convention, without changing populated sums.
    #[test]
    fn screw_partial_uv_provenance_defaults_missing_corners_to_canonical_uv() {
        let mut source = Mesh::from_positions_and_faces(
            vec![
                DVec3::ZERO,
                DVec3::X,
                DVec3::Y,
                DVec3::new(2.0, 0.0, 0.0),
                DVec3::new(3.0, 0.0, 0.0),
                DVec3::new(2.0, 1.0, 0.0),
            ],
            vec![vec![0, 1, 2], vec![3, 4, 5]],
        )
        .expect("two source faces");
        uv_layers::write_layers(
            &mut source,
            &[UvLayer {
                name: "UVMap".to_owned(),
                faces: vec![Some(vec![[2.0, 4.0], [4.0, 6.0], [6.0, 8.0]]), None],
                pinned: vec![None, None],
            }],
        );
        let mut output = Mesh::from_positions_and_faces(
            vec![
                DVec3::new(0.0, 0.0, 1.0),
                DVec3::new(1.0, 0.0, 1.0),
                DVec3::new(1.0, 1.0, 1.0),
                DVec3::new(0.0, 1.0, 1.0),
            ],
            vec![vec![0, 1, 2, 3]],
        )
        .expect("generated face");
        let origin = Origin {
            vertices: vec![
                vec![(0, 1.0)],
                vec![(3, 1.0)],
                vec![(1, 0.5), (3, 0.5)],
                Vec::new(),
            ],
            source_vertices: None,
            faces: vec![None],
            corners: Some(vec![
                vec![(0, 1.0)],
                vec![(3, 1.0)],
                vec![(1, 0.5), (3, 0.5)],
                Vec::new(),
            ]),
            ..Default::default()
        };

        remap_attributes(&source, &mut output, &origin).expect("remap");

        assert_eq!(
            uv_layers::read_layers(&output)[0].faces[0],
            Some(vec![[2.0, 4.0], [0.0, 1.0], [2.0, 3.5], [0.0, 1.0]])
        );
    }

    /// Bevel's point quaternion callbacks replay in order: a mutated source keeps its written value for
    /// later events, and an output vertex without an event copies its identity's stored value.
    #[test]
    fn quaternion_point_callbacks_replay_mutated_source_values() {
        let mut source = triangle();
        let input_ids: Vec<u32> = source.vertices.iter().map(|vertex| vertex.id).collect();
        let half = std::f32::consts::FRAC_1_SQRT_2;
        let three_quarter_turn = json!([half, 0.0, 0.0, -half]);
        let identity = json!([0.0, 0.0, 0.0, 1.0]);
        let small_rotation = json!([(0.0002_f32).sin(), 0.0, 0.0, (0.0002_f32).cos()]);
        source.attributes.insert(
            "orientation".to_owned(),
            json!({
                "domain": "point",
                "type": "quaternion",
                "values": keyed(vec![
                    (format!("v{}", input_ids[0]), three_quarter_turn),
                    (format!("v{}", input_ids[1]), identity),
                    (format!("v{}", input_ids[2]), small_rotation.clone()),
                ]),
            }),
        );
        let mut output = triangle();
        let origin = Origin {
            vertices: vec![vec![(0, 1.0)], vec![(0, 0.25), (1, 0.75)], vec![(2, 1.0)]],
            faces: vec![Some(0)],
            point_interpolation_events: Some(vec![
                PointInterpolationEvent {
                    output_vertex: 0,
                    mutated_source_vertex: Some(0),
                    sources: vec![(0, 1.0)],
                },
                PointInterpolationEvent {
                    output_vertex: 1,
                    mutated_source_vertex: None,
                    sources: vec![(0, 0.25), (1, 0.75)],
                },
            ]),
            ..Default::default()
        };
        let point_identity = vec![0, 1, 2];
        remap_attributes_with_edge_identity(
            &source,
            &mut output,
            &origin,
            Some(point_identity.as_slice()),
        )
        .expect("remap");
        let values = &output.attributes["orientation"]["values"];
        let updated_source = values[format!("v{}", output.vertices[0].id)]
            .as_array()
            .expect("updated source quaternion");
        let half = f64::from(std::f32::consts::FRAC_1_SQRT_2);
        assert!((updated_source[0].as_f64().expect("x") + half).abs() < 1e-6);
        assert!((updated_source[3].as_f64().expect("w") - half).abs() < 1e-6);
        let mixed = values[format!("v{}", output.vertices[1].id)]
            .as_array()
            .expect("interpolated quaternion");
        let expected_x = (-std::f32::consts::PI / 16.0).sin();
        let expected_w = (-std::f32::consts::PI / 16.0).cos();
        assert!((mixed[0].as_f64().expect("x") - f64::from(expected_x)).abs() < 1e-6);
        assert!((mixed[3].as_f64().expect("w") - f64::from(expected_w)).abs() < 1e-6);
        assert_eq!(
            values[format!("v{}", output.vertices[2].id)],
            small_rotation
        );
    }

    /// Interpolated quaternion corners mix through the expmap; original corners copy their source value.
    #[test]
    fn quaternion_corner_callbacks_mix_one_hot_values_but_original_corners_copy() {
        let mut source = triangle();
        let face_id = source.faces[0].id;
        let half = std::f32::consts::FRAC_1_SQRT_2;
        let three_quarter_turn = json!([half, 0.0, 0.0, -half]);
        let identity = json!([0.0, 0.0, 0.0, 1.0]);
        let small_rotation = json!([(0.0002_f32).sin(), 0.0, 0.0, (0.0002_f32).cos()]);
        source.attributes.insert(
            "corner_orientation".to_owned(),
            json!({
                "domain": "corner",
                "type": "quaternion",
                "values": keyed(vec![(
                    format!("f{face_id}"),
                    json!([three_quarter_turn, identity.clone(), small_rotation.clone()]),
                )]),
            }),
        );
        let mut output = triangle();
        let origin = Origin {
            vertices: vec![vec![(0, 1.0)], vec![(1, 1.0)], vec![(2, 1.0)]],
            faces: vec![Some(0)],
            corner_interpolated: Some(vec![true, false, false]),
            corners: Some(vec![vec![(0, 1.0)], vec![(1, 1.0)], vec![(2, 1.0)]]),
            ..Default::default()
        };
        remap_attributes(&source, &mut output, &origin).expect("remap");
        let corners =
            &output.attributes["corner_orientation"]["values"][format!("f{}", output.faces[0].id)];
        let mixed = corners[0].as_array().expect("interpolated quaternion");
        let half = f64::from(std::f32::consts::FRAC_1_SQRT_2);
        assert!((mixed[0].as_f64().expect("x") + half).abs() < 1e-6);
        assert!((mixed[3].as_f64().expect("w") - half).abs() < 1e-6);
        assert_eq!(corners[1], identity);
        assert_eq!(corners[2], small_rotation);
    }

    /// FLOAT2 corner attributes merge their own source corners, independently of other FLOAT2 layers.
    #[test]
    fn float2_corner_attributes_use_independent_merged_sources() {
        let mut source = triangle();
        let source_face = source.faces[0].id;
        for (name, row) in [
            ("corner_vector", json!([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])),
            ("corner_other", json!([[0.0, 0.0], [0.0, 1.0], [1.0, 1.0]])),
        ] {
            source.attributes.insert(
                name.to_owned(),
                json!({
                    "domain": "corner",
                    "type": "float2",
                    "values": keyed(vec![(format!("f{source_face}"), row)]),
                }),
            );
        }
        let mut output = triangle();
        let origin = Origin {
            vertices: vec![vec![(0, 1.0)], vec![(1, 1.0)], vec![(2, 1.0)]],
            faces: vec![Some(0)],
            corner_float2_sources: Some(vec![
                CornerFloat2Sources {
                    key: CornerFloat2LayerKey::Attribute("corner_vector".to_owned()),
                    corners: vec![vec![(0, 0.25), (1, 0.75)], vec![(1, 1.0)], vec![(2, 1.0)]],
                },
                CornerFloat2Sources {
                    key: CornerFloat2LayerKey::Attribute("corner_other".to_owned()),
                    corners: vec![vec![(0, 1.0)], vec![(1, 1.0)], vec![(2, 1.0)]],
                },
            ]),
            corners: Some(vec![vec![(0, 1.0)], vec![(1, 1.0)], vec![(2, 1.0)]]),
            ..Default::default()
        };
        remap_attributes(&source, &mut output, &origin).expect("remap");
        let face_key = format!("f{}", output.faces[0].id);
        assert_eq!(
            output.attributes["corner_vector"]["values"][face_key.as_str()][0],
            json!([0.75, 0.0])
        );
        assert_eq!(
            output.attributes["corner_other"]["values"][face_key.as_str()][0],
            json!([0.0, 0.0])
        );
    }

    /// UV FLOAT2 data merges connected loop sources, while each loop's pin follows its own
    /// unmerged source corner.
    #[test]
    fn uv_pins_follow_unmerged_source_corners() {
        let mut source = triangle();
        uv_layers::write_layers(
            &mut source,
            &[UvLayer {
                name: "UVMap".to_owned(),
                faces: vec![Some(vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])],
                pinned: vec![Some(vec![true, false, false])],
            }],
        );
        let mut output = triangle();
        let origin = Origin {
            vertices: vec![vec![(0, 1.0)], vec![(1, 1.0)], vec![(2, 1.0)]],
            faces: vec![Some(0)],
            corner_float2_sources: Some(vec![CornerFloat2Sources {
                key: CornerFloat2LayerKey::Uv("UVMap".to_owned()),
                corners: vec![
                    vec![(0, 0.5), (1, 0.5)],
                    vec![(0, 0.5), (1, 0.5)],
                    vec![(2, 1.0)],
                ],
            }]),
            corners: Some(vec![vec![(0, 1.0)], vec![(1, 1.0)], vec![(2, 1.0)]]),
            ..Default::default()
        };
        remap_attributes(&source, &mut output, &origin).expect("remap");
        let layers = uv_layers::read_layers(&output);
        assert_eq!(
            layers[0].faces[0],
            Some(vec![[0.5, 0.0], [0.5, 0.0], [0.0, 1.0]])
        );
        assert_eq!(layers[0].pinned[0], Some(vec![true, false, false]));
    }
}
