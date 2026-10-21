use std::collections::{BTreeMap, BTreeSet};

use glam::{DMat4, DVec3};
use serde::Serialize;
use serde_json::Value;

use crate::{
    error::{ErrorCode, PotError, Result},
    geom::{
        Mesh, attribute_keys, color_layers,
        uv_layers::{self, UvLayer, UvStore},
    },
    model::{DataBlock, Id, Node, SceneDoc, Transform, VertexGroup},
};

#[derive(Debug, Clone, Serialize)]
pub struct Loss {
    pub feature_id: String,
    pub data_id: Option<String>,
    pub reason: String,
    pub suggestion: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ExportReport {
    pub files: Vec<std::path::PathBuf>,
    pub counts: serde_json::Value,
    pub conversions: Vec<serde_json::Value>,
    pub losses: Vec<Loss>,
}

pub struct ImportedGraph {
    pub doc: SceneDoc,
    pub losses: Vec<Loss>,
    pub id_mappings: serde_json::Value,
    pub compat_blobs: Vec<(String, Vec<u8>)>,
    pub assets: Vec<(String, Vec<u8>)>,
    pub source: serde_json::Value,
}

pub(crate) fn is_default_material_graph(doc: &SceneDoc, graph_id: &Id) -> bool {
    doc.node_groups
        .get(graph_id)
        .is_some_and(crate::graph::is_generated_simple_material_graph)
}

pub struct ExportOptions<'a> {
    pub allow_lossy: bool,
    pub pack: bool,
    pub blender: Option<&'a std::path::Path>,
    pub context: &'a crate::eval::EvaluationContext,
}

/// Geometry for a single exported node in world-space coordinates.
#[derive(Clone, Debug)]
pub(crate) struct ExchangeMesh {
    pub id: String,
    pub name: String,
    pub positions: Vec<[f64; 3]>,
    pub faces: Vec<Vec<usize>>,
    /// World-space shading normals, one per face corner in `faces` order. `None` when the
    /// source format carries no normals.
    pub normals: Option<Vec<[f64; 3]>>,
    /// Whether the node's world transform mirrors geometry. Writers that bake world space
    /// reverse winding for these meshes; writers that keep the transform do not.
    pub mirrored: bool,
    /// Whether the node's world transform is singular. OBJ writes zero normals for these nodes,
    /// as Blender does.
    pub singular: bool,
    /// World-space per-vertex normals, indexed like `positions`: Blender's PLY vertex normals.
    /// Empty for imported meshes.
    pub vertex_normals: Vec<[f64; 3]>,
    /// Per face: whether a corner normal differs from the face's flat normal, in local space.
    /// Empty for imported meshes, where every face counts as flat.
    pub smooth: Vec<bool>,
    /// Imported `sharp_face` flags, one per face in `faces` order, as the format's Blender reader creates
    /// them. `None` when the reader creates no `sharp_face`, so every face stays smooth. Unused by exports.
    pub sharp_faces: Option<Vec<bool>>,
    /// UV layers in mesh order: glTF `TEXCOORD_n` and Blender's layer order.
    pub uv_layers: Vec<ExchangeUvLayer>,
    /// Index into `uv_layers` of the active (edit target) layer. OBJ and PLY write this layer.
    pub uv_active: Option<usize>,
    /// Index into `uv_layers` of the render layer.
    pub uv_render: Option<usize>,
    /// Canonical linear RGBA color layers, in source vertex/corner storage order.
    pub colors: color_layers::ColorStore,
    /// Sparse evaluated vertex-group assignments. A stored zero weight is an assignment.
    pub vertex_groups: Vec<ExchangeVertexGroup>,
    /// Resolved material slot ids from `Snapshot::slot_material`, one entry per slot.
    pub material_slots: Vec<Option<Id>>,
    /// Evaluated face slot indices, aligned with `faces`.
    pub face_material_indices: Vec<u32>,
}

/// One named vertex group, represented by sparse `(source vertex index, weight)` assignments.
#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct ExchangeVertexGroup {
    pub name: String,
    pub weights: Vec<(usize, f64)>,
}

/// Converts one canonical color layer to Blender's point-domain lookup result. Corner values are
/// averaged over every incident face corner; vertices with no incident corners use zero RGBA.
pub(super) fn point_color_values(
    mesh: &ExchangeMesh,
    layer: &color_layers::ColorLayer,
) -> Result<Vec<[f64; 4]>> {
    if layer
        .values
        .iter()
        .flatten()
        .any(|value| !value.is_finite())
    {
        return Err(PotError::new(
            ErrorCode::SceneInvalid,
            "color layer contains a non-finite value",
        ));
    }
    match layer.domain {
        color_layers::ColorDomain::Point => {
            if layer.values.len() != mesh.positions.len() {
                return Err(PotError::new(
                    ErrorCode::SceneInvalid,
                    "point color layer does not match vertex count",
                ));
            }
            Ok(layer.values.clone())
        }
        color_layers::ColorDomain::Corner => {
            let corner_count = mesh
                .faces
                .iter()
                .try_fold(0_usize, |count, face| count.checked_add(face.len()));
            let Some(corner_count) = corner_count else {
                return Err(PotError::new(
                    ErrorCode::LimitExceeded,
                    "corner color layer size overflow",
                ));
            };
            if layer.values.len() != corner_count {
                return Err(PotError::new(
                    ErrorCode::SceneInvalid,
                    "corner color layer does not match face corners",
                ));
            }
            let mut sums = vec![[0.0_f32; 4]; mesh.positions.len()];
            let mut counts = vec![0_usize; mesh.positions.len()];
            let mut corner = 0;
            for face in &mesh.faces {
                for &vertex in face {
                    let Some(sum) = sums.get_mut(vertex) else {
                        return Err(PotError::new(
                            ErrorCode::SceneInvalid,
                            "face references a missing vertex",
                        ));
                    };
                    // Blender mixes CORNER colours in float32 (`ColorGeometry4fMixer::mix_in`).
                    for (sum, value) in sum.iter_mut().zip(layer.values[corner]) {
                        *sum += value as f32;
                    }
                    counts[vertex] += 1;
                    corner += 1;
                }
            }
            for (sum, count) in sums.iter_mut().zip(counts) {
                if count != 0 {
                    let count = u32::try_from(count).map_err(|_| {
                        PotError::new(
                            ErrorCode::LimitExceeded,
                            "vertex corner count exceeds color average range",
                        )
                    })?;
                    // Blender multiplies by the float32 reciprocal of the total weight (`finalize`).
                    let inverse = 1.0_f32 / count as f32;
                    for value in sum {
                        *value *= inverse;
                    }
                }
            }
            Ok(sums.into_iter().map(|sum| sum.map(f64::from)).collect())
        }
    }
}

/// One UV layer of an exchange mesh.
#[derive(Clone, Debug)]
pub(crate) struct ExchangeUvLayer {
    pub name: String,
    /// Per face, the corner `[u, v]` values in `faces` order, in Potter's storage convention (v
    /// down). A face without an entry holds `[0, 0]` per corner.
    pub corners: Vec<Vec<[f64; 2]>>,
    /// Per-corner pin flags in the same face/corner order. Blender's exporters write none.
    pub pinned: Vec<Vec<bool>>,
}

/// Corner values of `layer` for each face of `faces`, in storage convention.
fn uv_corners(layer: &UvLayer, faces: &[crate::geom::Face]) -> Vec<Vec<[f64; 2]>> {
    faces
        .iter()
        .enumerate()
        .map(|(index, face)| {
            let entry = layer.faces.get(index).and_then(Option::as_ref);
            (0..face.vertices.len())
                .map(|corner| {
                    entry
                        .and_then(|values| values.get(corner))
                        .copied()
                        .unwrap_or([0.0, 0.0])
                })
                .collect()
        })
        .collect()
}

/// Per-corner pin flags of `layer` for each face of `faces`, in the same order as `uv_corners`.
fn uv_pins(layer: &UvLayer, faces: &[crate::geom::Face]) -> Vec<Vec<bool>> {
    faces
        .iter()
        .enumerate()
        .map(|(index, face)| {
            let entry = layer.pinned.get(index).and_then(Option::as_ref);
            (0..face.vertices.len())
                .map(|corner| {
                    entry
                        .and_then(|flags| flags.get(corner))
                        .copied()
                        .unwrap_or(false)
                })
                .collect()
        })
        .collect()
}

/// Writes imported UV layers into `mesh`. Layer `k` takes the corners of `layers[k]`, keyed by
/// storage face index, which `Mesh::from_positions_and_faces` keeps in `faces` order. `active` and
/// `render` index into `layers`.
pub(super) fn import_uv_layers(
    mesh: &mut Mesh,
    layers: &[ExchangeUvLayer],
    active: Option<usize>,
    render: Option<usize>,
) {
    let name_of = |index: Option<usize>| {
        index
            .and_then(|index| layers.get(index))
            .map(|layer| layer.name.clone())
    };
    let store = UvStore {
        layers: layers
            .iter()
            .map(|layer| {
                UvLayer::new(
                    layer.name.clone(),
                    layer.corners.iter().cloned().map(Some).collect(),
                )
            })
            .collect(),
        active: name_of(active),
        render: name_of(render),
    };
    uv_layers::write(mesh, &store);
}

/// A UV layer name an importer can store: `raw` when it is non-empty and unused, else `UVMap`, or
/// `raw` with a Blender-style `.001` suffix when it is taken.
pub(super) fn unique_uv_name(raw: &str, taken: &[ExchangeUvLayer]) -> String {
    let base = if raw.is_empty() { "UVMap" } else { raw };
    let is_taken = |candidate: &str| taken.iter().any(|layer| layer.name == candidate);
    if !is_taken(base) {
        return base.to_owned();
    }
    let mut suffix = 1_u32;
    loop {
        let candidate = format!("{base}.{suffix:03}");
        if !is_taken(&candidate) {
            return candidate;
        }
        suffix += 1;
    }
}

/// Loss for a UV layer an importer cannot represent.
pub(super) fn uv_import_loss(data_id: &str, reason: impl Into<String>) -> Loss {
    Loss {
        feature_id: "mesh.uv_layer".to_owned(),
        data_id: Some(data_id.to_owned()),
        reason: reason.into(),
        suggestion: Some("export as usda, usdc, fbx, or alembic to keep every UV layer".to_owned()),
    }
}

/// The one UV set a file format carries, as the layer `UVMap`, which is both active and render.
/// `corners` is per face in storage convention, aligned with the mesh faces.
pub(super) fn single_uv_set(
    corners: Vec<Vec<[f64; 2]>>,
) -> (Vec<ExchangeUvLayer>, Option<usize>, Option<usize>) {
    let pinned = corners.iter().map(|face| vec![false; face.len()]).collect();
    (
        vec![ExchangeUvLayer {
            name: "UVMap".to_owned(),
            corners,
            pinned,
        }],
        Some(0),
        Some(0),
    )
}

/// Loss entries for UV layers a format does not write. OBJ and PLY write only the active layer.
/// STL has no UV coordinates, so every layer is lost. Alembic omits the layers whose names it
/// reserves. Other formats keep every layer.
pub(crate) fn uv_layer_losses(format: &str, meshes: &[ExchangeMesh]) -> Vec<Loss> {
    let mut losses = Vec::new();
    for mesh in meshes {
        // Blender's OBJ, PLY, USD, FBX and Alembic exporters write no UV pin flags (its io sources
        // never read `pin_uv`), so a pinned layer loses them in every non-.blend format.
        for layer in &mesh.uv_layers {
            if layer.pinned.iter().flatten().any(|pinned| *pinned) {
                losses.push(Loss {
                    feature_id: "mesh.uv_layer".to_owned(),
                    data_id: Some(mesh.id.clone()),
                    reason: format!(
                        "{format} does not preserve per-corner UV pin flags for layer {}",
                        layer.name
                    ),
                    suggestion: Some("export as blend to keep UV pin flags".to_owned()),
                });
            }
        }
        for (index, layer) in mesh.uv_layers.iter().enumerate() {
            let (reason, suggestion) = match format {
                "obj" | "ply" if mesh.uv_active != Some(index) => (
                    format!(
                        "{format} stores only the active UV layer; UV layer `{}` is lost",
                        layer.name
                    ),
                    "export as usda, usdc, fbx, or alembic to keep every UV layer",
                ),
                "stl" => (
                    format!(
                        "stl stores no UV coordinates; UV layer `{}` is lost",
                        layer.name
                    ),
                    "export as obj, ply, usda, fbx, or alembic to keep UV coordinates",
                ),
                "alembic" if alembic::reserves_uv_layer_name(&layer.name) => (
                    format!(
                        "Alembic reserves the property name `{0}`; UV layer `{0}` is not written",
                        layer.name
                    ),
                    "rename the UV layer, or export as usda, usdc, or fbx to keep it",
                ),
                _ => continue,
            };
            losses.push(Loss {
                feature_id: "mesh.uv_layer".to_owned(),
                data_id: Some(mesh.id.clone()),
                reason,
                suggestion: Some(suggestion.to_owned()),
            });
        }
    }
    losses
}

pub(crate) fn evaluated_meshes(
    doc: &SceneDoc,
    snapshot: &crate::eval::Snapshot,
) -> Result<Vec<ExchangeMesh>> {
    let visible_nodes = render_visible_nodes(doc, snapshot)?;
    let mut output = Vec::new();
    for (node_id, node) in &doc.nodes {
        if !node.visible
            || !node.render_visible
            || node.data.is_none()
            || !visible_nodes.contains(node_id)
        {
            continue;
        }
        let Some(source) = snapshot.meshes.get(node_id) else {
            continue;
        };
        let matrix = snapshot
            .nodes
            .get(node_id)
            .map(|state| DMat4::from_cols_array(&state.world_matrix))
            .ok_or_else(|| PotError::new(ErrorCode::EvaluationFailed, "node evaluation missing"))?;
        let positions = source
            .vertices
            .iter()
            .map(|vertex| matrix.transform_point3(vertex.co).to_array())
            .collect::<Vec<_>>();
        let vertex_indices = source
            .vertices
            .iter()
            .enumerate()
            .map(|(index, vertex)| (vertex.id, index))
            .collect::<BTreeMap<_, _>>();
        let local = source
            .vertices
            .iter()
            .map(|vertex| vertex.co)
            .collect::<Vec<_>>();
        let shading = shading::world_normals(source, &local, matrix)?;
        let mut faces = Vec::with_capacity(source.faces.len());
        for face in &source.faces {
            let Some(indices) = face
                .vertices
                .iter()
                .map(|id| vertex_indices.get(id).copied())
                .collect::<Option<Vec<_>>>()
            else {
                return Err(PotError::new(
                    ErrorCode::SceneInvalid,
                    "mesh face references a missing vertex",
                ));
            };
            faces.push(indices);
        }
        let uv = uv_layers::read(source);
        let uv_layer_list = uv
            .layers
            .iter()
            .map(|layer| ExchangeUvLayer {
                name: layer.name.clone(),
                corners: uv_corners(layer, &source.faces),
                pinned: uv_pins(layer, &source.faces),
            })
            .collect::<Vec<_>>();
        let uv_active = uv.active.as_deref().and_then(|name| uv.index_of(name));
        let uv_render = uv.render_index();
        let data = node.data.as_ref().and_then(|id| doc.data_blocks.get(id));
        let vertex_groups = data.map_or_else(Vec::new, |data| {
            let weights = crate::eval::mesh_vertex_weights(source, &data.vertex_groups);
            data.vertex_groups
                .iter()
                .map(|group| ExchangeVertexGroup {
                    name: group.name.clone(),
                    weights: source
                        .vertices
                        .iter()
                        .enumerate()
                        .filter_map(|(index, vertex)| {
                            weights
                                .get(&vertex.id)
                                .and_then(|by_group| by_group.get(&group.id))
                                .map(|weight| (index, *weight))
                        })
                        .collect(),
                })
                .collect()
        });
        let slot_count = snapshot.slot_count(node_id, node, data);
        output.push(ExchangeMesh {
            id: node_id.to_string(),
            name: node.name.clone(),
            positions,
            faces,
            normals: Some(shading.corners),
            vertex_normals: shading.vertices,
            smooth: shading.smooth,
            sharp_faces: None,
            singular: shading::is_singular(matrix),
            mirrored: shading::is_mirrored(matrix),
            uv_layers: uv_layer_list,
            uv_active,
            uv_render,
            colors: color_layers::read(source),
            vertex_groups,
            material_slots: (0..slot_count)
                .map(|slot| snapshot.slot_material(node_id, node, data, slot).cloned())
                .collect(),
            face_material_indices: source
                .faces
                .iter()
                .map(|face| face.material_index)
                .collect(),
        });
    }
    Ok(output)
}

/// Loss entries for the color layers and generic element attributes of the exported meshes that
/// `format` cannot write. Writers never receive these attributes, so the loss is the only record:
/// strict export refuses on it, and `--allow-lossy` permits it.
pub(crate) fn attribute_losses(
    format: &str,
    doc: &SceneDoc,
    snapshot: &crate::eval::Snapshot,
) -> Result<Vec<Loss>> {
    let visible_nodes = render_visible_nodes(doc, snapshot)?;
    let mut losses = Vec::new();
    for (node_id, node) in &doc.nodes {
        if !node.visible
            || !node.render_visible
            || node.data.is_none()
            || !visible_nodes.contains(node_id)
        {
            continue;
        }
        let Some(source) = snapshot.meshes.get(node_id) else {
            continue;
        };
        losses.extend(mesh_attribute_losses(format, &node_id.to_string(), source));
    }
    Ok(losses)
}

/// Losses for one evaluated mesh. Shading keys go through the shared shading path, whose format
/// limits `shading_losses` reports, and UV keys through `uv_layer_losses`. Dot-prefixed
/// `blender_attributes` are Blender selection and hide state, not mesh data, so they are not
/// reported. Every other Blender attribute is user data and is reported.
pub(crate) fn mesh_attribute_losses(format: &str, mesh_id: &str, mesh: &Mesh) -> Vec<Loss> {
    let color = color_layers::read(mesh);
    let mut losses = Vec::new();
    for key in mesh.attributes.keys() {
        // Potter's shading, UV, color, vertex-group, and Blender bookkeeping stores have their own
        // losses or writers. Other internal stores are not exchange data and are reported by key.
        let name = match attribute_keys::user_name(key) {
            Some(name) => name,
            None if matches!(
                key.as_str(),
                uv_layers::ATTRIBUTE
                    | color_layers::ATTRIBUTE
                    | "vertex_groups"
                    | "blender_attributes"
                    | "blender_metadata"
                    | "custom_normal"
                    | "shade_smooth"
                    | "sharp"
                    | crate::geom::modifiers::OPTIMAL_DISPLAY_EDGES
            ) || (format == "alembic" && key == "alembic_normals") =>
            {
                continue;
            }
            None => key.as_str(),
        };
        if color.layers.iter().any(|layer| layer.name == name) {
            continue;
        }
        losses.push(Loss {
            feature_id: "mesh.attribute".to_owned(),
            data_id: Some(mesh_id.to_owned()),
            reason: format!("{format} does not write mesh attribute `{name}`"),
            suggestion: Some("export as blend to keep mesh attributes".to_owned()),
        });
    }
    let blender_entries = mesh
        .attributes
        .get("blender_attributes")
        .and_then(Value::as_array)
        .into_iter()
        .flatten();
    for entry in blender_entries {
        let Some(name) = entry["name"].as_str().filter(|name| !name.starts_with('.')) else {
            continue;
        };
        let (feature_id, kind) = if gltf::is_color_attribute(entry) {
            ("mesh.color_attribute", "color attribute")
        } else {
            ("mesh.attribute", "attribute")
        };
        losses.push(Loss {
            feature_id: feature_id.to_owned(),
            data_id: Some(mesh_id.to_owned()),
            reason: format!("{format} does not write Blender {kind} `{name}`"),
            suggestion: Some("export as blend to keep mesh attributes".to_owned()),
        });
    }
    losses
}

/// Precise losses for canonical color layers, vertex-group assignments, and face-material
/// bindings that `format` cannot write. Writers never receive these, so these losses are the only
/// record: strict export refuses on them and `--allow-lossy` permits them.
pub(crate) fn element_data_losses(format: &str, meshes: &[ExchangeMesh]) -> Vec<Loss> {
    let mut losses = Vec::new();
    for mesh in meshes {
        let selected_color = mesh
            .colors
            .active
            .as_deref()
            .and_then(|name| mesh.colors.index_of(name));
        for (index, layer) in mesh.colors.layers.iter().enumerate() {
            let point_values = if matches!(format, "obj" | "ply") && selected_color == Some(index) {
                point_color_values(mesh, layer).ok()
            } else {
                None
            };
            let values = point_values.as_deref().unwrap_or(&layer.values);
            if format == "stl" {
                losses.push(element_loss(
                    mesh,
                    "mesh.color_attribute",
                    format!(
                        "stl has no color channel; color layer `{}` is lost",
                        layer.name
                    ),
                    "export as obj, ply, usda, usdc, fbx, or alembic to preserve colors",
                ));
            } else if matches!(format, "obj" | "ply") && selected_color != Some(index) {
                losses.push(element_loss(
                    mesh,
                    "mesh.color_attribute",
                    format!(
                        "{format} stores only the active color layer; color layer `{}` is lost",
                        layer.name
                    ),
                    "export as usda, usdc, fbx, or alembic to preserve every color layer",
                ));
            } else if matches!(format, "obj" | "ply") {
                let imported_name = if format == "obj" { "Color" } else { "Col" };
                if layer.name != imported_name {
                    losses.push(element_loss(
                        mesh,
                        "mesh.color_attribute",
                        format!(
                            "{format} imports the active color layer as `{imported_name}`; source layer `{}` is renamed",
                            layer.name
                        ),
                        "export as usda, usdc, fbx, or alembic to preserve color-layer names",
                    ));
                }
                if layer.domain == color_layers::ColorDomain::Corner {
                    losses.push(element_loss(
                        mesh,
                        "mesh.color_attribute",
                        format!(
                            "{format} averages CORNER layer `{}` to point colors; per-corner differences are lost",
                            layer.name
                        ),
                        "export as usda, usdc, fbx, or alembic to preserve the color domain",
                    ));
                }
                if mesh.colors.render.as_deref() != Some(layer.name.as_str()) {
                    losses.push(element_loss(
                        mesh,
                        "mesh.color_attribute",
                        format!(
                            "{format} does not preserve separate active/render color-layer designations for `{}`",
                            layer.name
                        ),
                        "export as usda, usdc, or fbx to preserve color-layer designations",
                    ));
                }
            }
            if format == "ply" && selected_color == Some(index) {
                let rgb_loss = values.iter().any(|rgba| {
                    rgba[..3].iter().any(|value| {
                        let decoded = ply_channel_roundtrip(*value, false);
                        !decoded.is_finite() || (decoded - *value).abs() > 1.0e-6
                    })
                });
                let rgb_clipped = values
                    .iter()
                    .any(|rgba| rgba[..3].iter().any(|value| !(0.0..=1.0).contains(value)));
                if rgb_loss {
                    let reason = if rgb_clipped {
                        format!(
                            "PLY's active RGB properties clip values outside [0, 1] and quantize color layer `{}` to 8-bit sRGB",
                            layer.name
                        )
                    } else {
                        format!(
                            "PLY's active RGB properties quantize color layer `{}` to 8-bit sRGB",
                            layer.name
                        )
                    };
                    losses.push(element_loss(
                        mesh,
                        "mesh.color_attribute",
                        reason,
                        "export as usda, usdc, or fbx to preserve float color precision and HDR",
                    ));
                }
                let alpha_loss = values.iter().any(|rgba| {
                    let decoded = ply_channel_roundtrip(rgba[3], true);
                    !decoded.is_finite() || (decoded - rgba[3]).abs() > 1.0e-6
                });
                if alpha_loss {
                    losses.push(element_loss(
                        mesh,
                        "mesh.color_attribute",
                        format!(
                            "PLY's active alpha property clips or quantizes color layer `{}` to 8-bit",
                            layer.name
                        ),
                        "export as usda, usdc, fbx, or alembic to preserve full-precision alpha",
                    ));
                }
            }
            if matches!(format, "fbx" | "fbx-binary" | "alembic") {
                let byte_precision_loss = layer.values.iter().any(|rgba| {
                    rgba.iter().enumerate().any(|(channel, value)| {
                        let decoded = byte_color_roundtrip(*value, channel == 3);
                        !decoded.is_finite() || (decoded - *value).abs() > 1.0e-6
                    })
                });
                if byte_precision_loss {
                    losses.push(element_loss(
                        mesh,
                        "mesh.color_attribute",
                        format!(
                            "{format} stores color layer `{}` as Blender 8-bit BYTE_COLOR; channel precision is quantized",
                            layer.name
                        ),
                        "export as usda or usdc to preserve float color precision",
                    ));
                }
                if layer
                    .values
                    .iter()
                    .flatten()
                    .any(|value| !(0.0..=1.0).contains(value))
                {
                    losses.push(element_loss(
                        mesh,
                        "mesh.color_attribute",
                        format!(
                            "{format} clips color layer `{}` channels outside [0, 1] for Blender BYTE_COLOR",
                            layer.name
                        ),
                        "export as usda or usdc to preserve HDR and out-of-range values",
                    ));
                }
                if layer.domain == color_layers::ColorDomain::Point {
                    let source_format = if format == "alembic" {
                        "Alembic"
                    } else {
                        "FBX"
                    };
                    losses.push(element_loss(
                        mesh,
                        "mesh.color_attribute",
                        format!(
                            "Blender imports {source_format} POINT color layer `{}` as CORNER data",
                            layer.name
                        ),
                        "export as blend, usda, or usdc to preserve the point domain",
                    ));
                }
            }
            if format == "obj" && selected_color == Some(index) {
                #[expect(
                    clippy::float_cmp,
                    reason = "exact 1.0 marks opaque alpha; any other alpha is non-opaque"
                )]
                if values.iter().any(|value| value[3] != 1.0) {
                    losses.push(element_loss(
                        mesh,
                        "mesh.color_attribute",
                        format!(
                            "OBJ xyzrgb stores RGB only; non-opaque alpha in color layer `{}` is lost",
                            layer.name
                        ),
                        "export as ply, usda, usdc, fbx, or alembic to preserve alpha",
                    ));
                }
                if values
                    .iter()
                    .any(|value| value[..3].iter().any(|component| *component < 0.0))
                {
                    losses.push(element_loss(
                        mesh,
                        "mesh.color_attribute",
                        format!(
                            "OBJ clamps negative RGB components in color layer `{}` to zero",
                            layer.name
                        ),
                        "export as usda, usdc, fbx, or alembic to preserve negative values",
                    ));
                }
            }
        }
        if format == "alembic"
            && let Some(first_name) = mesh.colors.layers.first().map(|layer| layer.name.as_str())
            && (mesh.colors.active.as_deref() != Some(first_name)
                || mesh.colors.render.as_deref() != Some(first_name))
        {
            losses.push(element_loss(
                mesh,
                "mesh.color_attribute",
                format!(
                    "Alembic preserves color-layer names but imports active and render selections as the first layer `{first_name}`"
                ),
                "export as blend, usda, usdc, or fbx to preserve color-layer selections",
            ));
        }
        if matches!(format, "alembic" | "ply" | "stl")
            && !mesh.material_slots.is_empty()
            && mesh.face_material_indices.iter().any(|face_slot| {
                let slot = usize::try_from(*face_slot)
                    .unwrap_or(usize::MAX)
                    .min(mesh.material_slots.len() - 1);
                mesh.material_slots[slot].is_some()
            })
        {
            losses.push(element_loss(
                mesh,
                "material.binding",
                format!("{format} does not preserve assigned face-material bindings"),
                "export as glb, obj, fbx, usda, usdc, or blend to preserve face materials",
            ));
        }
        for group in &mesh.vertex_groups {
            let reason = match format {
                "obj" if !obj_group_is_face_compatible(mesh, group) => Some(format!(
                    "OBJ face groups cannot represent sparse or non-binary weights in vertex group `{}`",
                    group.name
                )),
                "ply" | "stl" | "alembic" => Some(format!(
                    "{format} has no Blender-readable vertex-weight channel for group `{}`",
                    group.name
                )),
                _ => None,
            };
            if let Some(reason) = reason {
                losses.push(element_loss(
                    mesh,
                    "mesh.vertex_group",
                    reason,
                    "export as usda or fbx to preserve vertex-group weights",
                ));
            }
        }
        if format == "obj" {
            let compatible = mesh
                .vertex_groups
                .iter()
                .enumerate()
                .filter(|(_, group)| obj_group_is_face_compatible(mesh, group))
                .collect::<Vec<_>>();
            let mut dropped = BTreeSet::new();
            for face in &mesh.faces {
                if face.len() < 3 {
                    continue;
                }
                let mut matched = false;
                for (group_index, group) in &compatible {
                    if face
                        .iter()
                        .all(|vertex| group.weights.iter().any(|(index, _)| index == vertex))
                    {
                        if matched {
                            dropped.insert(*group_index);
                        } else {
                            matched = true;
                        }
                    }
                }
            }
            for group_index in dropped {
                let name = &mesh.vertex_groups[group_index].name;
                losses.push(element_loss(
                    mesh,
                    "mesh.vertex_group",
                    format!(
                        "OBJ writes one group per face; overlapping membership in vertex group `{name}` is not written"
                    ),
                    "export as usda or fbx to preserve overlapping vertex groups",
                ));
            }
        }
    }
    losses
}

fn element_loss(mesh: &ExchangeMesh, feature_id: &str, reason: String, suggestion: &str) -> Loss {
    Loss {
        feature_id: feature_id.to_owned(),
        data_id: Some(mesh.id.clone()),
        reason,
        suggestion: Some(suggestion.to_owned()),
    }
}

/// Linear light to sRGB-encoded, unclamped (the OBJ and PLY colour encoders use it).
pub(super) fn srgb_encode(value: f64) -> f64 {
    crate::color::linear_to_srgb_unclamped(value)
}

/// sRGB-encoded to linear light, the inverse of [`srgb_encode`].
pub(super) fn srgb_decode(value: f64) -> f64 {
    crate::color::srgb_to_linear(value)
}
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "the input is clamped and rounded to Blender's 8-bit channel range"
)]
pub(super) fn blender_color_byte(value: f64) -> u8 {
    // Blender rounds in float32 (`0.9f * 255 + 0.5` is 230.0, where float64 gives 229.99999).
    (value.clamp(0.0, 1.0) as f32 * 255.0_f32 + 0.5_f32) as u8
}

/// A canonical linear channel as Blender stores it in `BYTE_COLOR` interchange: encoded RGB, linear
/// alpha, quantized to the nearest byte and normalized back to `[0, 1]`.
pub(super) fn encode_byte_color(value: f64, alpha: bool) -> f64 {
    let encoded = if alpha { value } else { srgb_encode(value) };
    f64::from(blender_color_byte(encoded)) / 255.0
}

/// A normalized Blender `BYTE_COLOR` channel decoded to Potter's linear representation.
pub(super) fn decode_byte_color(value: f64, alpha: bool) -> f64 {
    let encoded = f64::from(blender_color_byte(value)) / 255.0;
    if alpha { encoded } else { srgb_decode(encoded) }
}

/// The canonical linear value represented by one Blender `BYTE_COLOR` round-trip.
pub(super) fn byte_color_roundtrip(value: f64, alpha: bool) -> f64 {
    decode_byte_color(encode_byte_color(value, alpha), alpha)
}
/// PLY stores colours as bytes: RGB is sRGB-encoded, alpha is linear. Clamps to `[0, 1]`.
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "encoded channels are clamped to [0, 1] before scaling into the byte range"
)]
pub(super) fn ply_color_to_byte(value: f64, alpha: bool) -> u8 {
    let encoded = if alpha { value } else { srgb_encode(value) };
    (encoded.clamp(0.0, 1.0) * 255.0).floor() as u8
}

/// The linear value a PLY colour byte stands for, the inverse of [`ply_color_to_byte`].
pub(super) fn ply_byte_to_color(value: u8, alpha: bool) -> f64 {
    let encoded = f64::from(value) / 255.0;
    if alpha { encoded } else { srgb_decode(encoded) }
}

fn ply_channel_roundtrip(value: f64, alpha: bool) -> f64 {
    ply_byte_to_color(ply_color_to_byte(value, alpha), alpha)
}

/// Whether a vertex group is a face-compatible OBJ group: binary weights on whole faces only.
pub(crate) fn obj_group_is_face_compatible(
    mesh: &ExchangeMesh,
    group: &ExchangeVertexGroup,
) -> bool {
    if group.weights.is_empty()
        || group.name.trim() != group.name.as_str()
        || group.name.contains('\n')
        || group.name.contains('\r')
        || group.name.contains('#')
    {
        return false;
    }
    let mut assigned = vec![false; mesh.positions.len()];
    for (index, weight) in &group.weights {
        let Some(value) = assigned.get_mut(*index) else {
            return false;
        };
        #[expect(
            clippy::float_cmp,
            reason = "exact 1.0 is the binary whole-face weight; other weights are rejected"
        )]
        if *weight != 1.0 || *value {
            return false;
        }
        *value = true;
    }
    mesh.faces.iter().all(|face| {
        let mut membership = None;
        face.iter().all(|index| {
            let Some(value) = assigned.get(*index) else {
                return false;
            };
            if let Some(expected) = membership {
                *value == expected
            } else {
                membership = Some(*value);
                true
            }
        })
    }) && group
        .weights
        .iter()
        .all(|(index, _)| mesh.faces.iter().any(|face| face.contains(index)))
}

pub(crate) fn render_visible_nodes(
    doc: &SceneDoc,
    snapshot: &crate::eval::Snapshot,
) -> Result<BTreeSet<Id>> {
    let scene = doc
        .scenes
        .get(&snapshot.scene_id)
        .ok_or_else(|| PotError::new(ErrorCode::EvaluationFailed, "evaluation scene is missing"))?;
    let excluded = snapshot
        .view_layer
        .as_ref()
        .and_then(|id| scene.view_layers.get(id))
        .map(|layer| {
            layer
                .excluded_collections
                .iter()
                .cloned()
                .collect::<BTreeSet<_>>()
        })
        .unwrap_or_default();
    let mut visible = BTreeSet::new();
    let mut visited = BTreeSet::new();
    let mut pending = vec![scene.root_collection.clone()];
    while let Some(id) = pending.pop() {
        if excluded.contains(&id) || !visited.insert(id.clone()) {
            continue;
        }
        let collection = doc.collections.get(&id).ok_or_else(|| {
            PotError::new(ErrorCode::SceneInvalid, "collection reference is missing")
        })?;
        visible.extend(collection.objects.iter().cloned());
        pending.extend(collection.children.iter().cloned());
    }
    Ok(visible)
}

pub(crate) fn graph_from_meshes(meshes: Vec<ExchangeMesh>, scene_id: String) -> Result<SceneDoc> {
    if meshes.is_empty() {
        return Err(PotError::new(
            ErrorCode::ImportFailed,
            "input contains no mesh geometry",
        ));
    }
    let mut doc = SceneDoc::new(scene_id);
    let mut imported_nodes = Vec::with_capacity(meshes.len());
    for (index, geometry) in meshes.into_iter().enumerate() {
        let stem = valid_stem(&geometry.id, index);
        let node_id = unique_graph_id(
            &doc,
            &stem,
            index,
            GraphIdRegistry::Nodes,
            "imported ID exceeds maximum length",
        )?;
        let data_id = unique_graph_id(
            &doc,
            &format!("{stem}_mesh"),
            index,
            GraphIdRegistry::DataBlocks,
            "imported data ID exceeds maximum length",
        )?;
        let positions = geometry
            .positions
            .iter()
            .map(|position| DVec3::from_array(*position))
            .collect();
        let mut mesh =
            Mesh::from_positions_and_faces(positions, geometry.faces).map_err(|error| {
                PotError::with_details(
                    ErrorCode::ImportFailed,
                    "imported geometry is invalid",
                    serde_json::json!({ "reason": error.to_string(), "data_id": data_id }),
                )
            })?;
        let normals = geometry.normals.as_ref().map(|normals| {
            normals
                .iter()
                .map(|normal| DVec3::from_array(*normal))
                .collect::<Vec<_>>()
        });
        shading::import_reader_shading(
            &mut mesh,
            normals.as_deref(),
            geometry.sharp_faces.as_deref(),
        )?;
        import_uv_layers(
            &mut mesh,
            &geometry.uv_layers,
            geometry.uv_active,
            geometry.uv_render,
        );
        color_layers::write(&mut mesh, &geometry.colors);
        let mut vertex_groups = Vec::with_capacity(geometry.vertex_groups.len());
        let mut vertex_weights = BTreeMap::new();
        for (group_index, exchange_group) in geometry.vertex_groups.iter().enumerate() {
            let group_id = Id::new(format!("vertex_group_{group_index}"))?;
            for &(vertex_index, weight) in &exchange_group.weights {
                let vertex = mesh.vertices.get(vertex_index).ok_or_else(|| {
                    PotError::new(
                        ErrorCode::ImportFailed,
                        "imported vertex group references a missing vertex",
                    )
                })?;
                vertex_weights
                    .entry(vertex.id)
                    .or_insert_with(BTreeMap::new)
                    .insert(group_id.clone(), weight);
            }
            vertex_groups.push(VertexGroup {
                id: group_id,
                name: exchange_group.name.clone(),
            });
        }
        doc.data_blocks.insert(
            data_id.clone(),
            DataBlock {
                data_type: "mesh".to_owned(),
                descriptor: None,
                mesh: Some(mesh),
                vertex_groups,
                vertex_weights,
                camera: None,
                light: None,
                ..DataBlock::default()
            },
        );
        doc.nodes.insert(
            node_id.clone(),
            Node {
                name: geometry.name,
                kind: "mesh".to_owned(),
                primitive: None,
                tags: Vec::new(),
                parent: None,
                parent_inverse: None,
                transform: Transform::default(),
                data: Some(data_id),
                materials: Vec::new(),
                modifiers: Vec::new(),
                visible: true,
                render_visible: true,
                selectable: true,
                action: None,
                properties: serde_json::Map::new(),
                ..Node::default()
            },
        );
        imported_nodes.push(node_id);
    }
    let root = Id::new("collection_root")?;
    doc.collections
        .get_mut(&root)
        .ok_or_else(|| PotError::new(ErrorCode::InternalError, "root collection is missing"))?
        .objects
        .extend(imported_nodes);
    doc.validate()?;
    Ok(doc)
}

pub(crate) fn import_graph_from_meshes(
    meshes: Vec<ExchangeMesh>,
    path: &std::path::Path,
    scene_id: String,
    format: &str,
    losses: Vec<Loss>,
) -> Result<ImportedGraph> {
    let source = serde_json::json!({ "format": format, "path": path.display().to_string() });
    Ok(ImportedGraph {
        doc: graph_from_meshes(meshes, scene_id)?,
        losses,
        id_mappings: serde_json::json!({}),
        compat_blobs: Vec::new(),
        assets: Vec::new(),
        source,
    })
}

fn valid_stem(value: &str, index: usize) -> String {
    let mut result = String::with_capacity(value.len() + 16);
    for byte in value.bytes() {
        let character = char::from(byte).to_ascii_lowercase();
        if character.is_ascii_lowercase()
            || character.is_ascii_digit()
            || character == '_'
            || character == '-'
        {
            result.push(character);
        } else {
            result.push('_');
        }
    }
    if result.is_empty() || !result.as_bytes()[0].is_ascii_lowercase() {
        result.insert_str(0, &format!("mesh{index}_"));
    }
    result.truncate(48);
    result
}

#[derive(Clone, Copy)]
enum GraphIdRegistry {
    Nodes,
    DataBlocks,
}

fn unique_graph_id(
    doc: &SceneDoc,
    seed: &str,
    index: usize,
    registry: GraphIdRegistry,
    too_long_message: &str,
) -> Result<Id> {
    let occupied = |id: &Id| match registry {
        GraphIdRegistry::Nodes => doc.nodes.contains_key(id),
        GraphIdRegistry::DataBlocks => doc.data_blocks.contains_key(id),
    };
    let mut candidate = seed.to_owned();
    let mut suffix = index;
    while occupied(&Id::new(candidate.clone())?) {
        suffix = suffix.saturating_add(1);
        candidate = format!("{seed}_{suffix}");
        if candidate.len() > 64 {
            return Err(PotError::new(ErrorCode::LimitExceeded, too_long_message));
        }
    }
    Id::new(candidate)
}

pub(super) fn claim_import_id(id: Id, used: &mut BTreeSet<Id>) -> Option<Id> {
    used.insert(id.clone()).then_some(id)
}

pub(super) fn import_error(message: impl Into<String>) -> PotError {
    PotError::new(ErrorCode::ImportFailed, message)
}

#[derive(Clone, Copy)]
pub(super) struct LeReadErrors {
    pub offset_overflow: &'static str,
    pub out_of_bounds: &'static str,
    pub invalid_integer: &'static str,
}

fn read_le_array<const N: usize>(
    bytes: &[u8],
    offset: usize,
    errors: LeReadErrors,
) -> Result<[u8; N]> {
    let end = offset
        .checked_add(N)
        .ok_or_else(|| import_error(errors.offset_overflow))?;
    let bytes = bytes
        .get(offset..end)
        .ok_or_else(|| import_error(errors.out_of_bounds))?;
    bytes
        .try_into()
        .map_err(|_| import_error(errors.invalid_integer))
}

pub(super) fn read_u32(bytes: &[u8], offset: usize, errors: LeReadErrors) -> Result<u32> {
    Ok(u32::from_le_bytes(read_le_array(bytes, offset, errors)?))
}

pub(super) fn read_u64(bytes: &[u8], offset: usize, errors: LeReadErrors) -> Result<u64> {
    Ok(u64::from_le_bytes(read_le_array(bytes, offset, errors)?))
}

pub(super) fn decode_le_array<const N: usize>(
    bytes: &[u8],
    invalid_message: &'static str,
) -> Result<[u8; N]> {
    bytes.try_into().map_err(|_| import_error(invalid_message))
}

pub(super) fn unique_data_id(
    first: Id,
    is_used: impl Fn(&Id) -> bool,
    mut with_suffix: impl FnMut(u32) -> Result<Id>,
    first_suffix: u32,
    exhausted_message: &'static str,
) -> Result<Id> {
    if !is_used(&first) {
        return Ok(first);
    }
    let mut suffix = first_suffix;
    loop {
        let candidate = with_suffix(suffix)?;
        if !is_used(&candidate) {
            return Ok(candidate);
        }
        suffix = suffix
            .checked_add(1)
            .ok_or_else(|| PotError::new(ErrorCode::LimitExceeded, exhausted_message))?;
    }
}

pub mod alembic;
pub mod blend;
mod blend_dna;
pub mod bvh;
pub mod fbx;
pub mod fbx_binary;
pub mod gltf;
pub(crate) mod material_graph;
pub(crate) mod material_surface;
pub mod obj;
pub mod pdf;
pub mod ply;
pub(crate) mod shading;
pub mod stl;
pub mod svg;
pub mod usd;
pub mod usdc;
mod vector;
pub mod vrm;

#[cfg(test)]
pub(crate) mod test_support {
    use super::{ExchangeMesh, ExchangeUvLayer};

    /// One triangle with two UV layers. Layer 0 is `UVMap` and is the render layer. Layer 1 is
    /// `Lightmap` and is the active layer, which OBJ and PLY write.
    pub(crate) fn triangle_with_layers() -> ExchangeMesh {
        use crate::geom::color_layers::ColorStore;

        ExchangeMesh {
            id: "triangle".to_owned(),
            name: "Triangle".to_owned(),
            positions: vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            faces: vec![vec![0, 1, 2]],
            normals: Some(vec![[0.0, 0.0, 1.0]; 3]),
            mirrored: false,
            singular: false,
            vertex_normals: Vec::new(),
            smooth: vec![false],
            sharp_faces: None,
            uv_layers: vec![
                ExchangeUvLayer {
                    name: "UVMap".to_owned(),
                    corners: vec![vec![[0.1, 0.25], [0.5, 0.75], [0.9, 0.0]]],
                    pinned: vec![vec![false; 3]],
                },
                ExchangeUvLayer {
                    name: "Lightmap".to_owned(),
                    corners: vec![vec![[0.2, 0.1], [0.6, 0.3], [0.4, 0.0]]],
                    pinned: vec![vec![false; 3]],
                },
            ],
            uv_active: Some(1),
            uv_render: Some(0),
            colors: ColorStore::default(),
            vertex_groups: Vec::new(),
            material_slots: Vec::new(),
            face_material_indices: vec![0],
        }
    }

    /// One triangle with color layers `Col` (point, active and render) and `Corner` (corner), a
    /// point float attribute `weight`, and `blender_attributes`: a byte color `Blender Col`, a user
    /// int `Pinned`, and the selection internal `.select_vert`.
    pub(crate) fn triangle_with_attributes() -> crate::error::Result<crate::geom::Mesh> {
        use crate::geom::color_layers::{self, ColorDomain, ColorLayer, ColorStore};
        use glam::DVec3;
        use serde_json::json;

        let mut mesh = crate::geom::Mesh::from_positions_and_faces(
            vec![DVec3::ZERO, DVec3::X, DVec3::Y],
            vec![vec![0, 1, 2]],
        )
        .map_err(|error| {
            crate::error::PotError::new(crate::error::ErrorCode::InternalError, error.to_string())
        })?;
        color_layers::write(
            &mut mesh,
            &ColorStore {
                layers: vec![
                    ColorLayer {
                        name: "Col".to_owned(),
                        domain: ColorDomain::Point,
                        values: vec![[1.0, 0.5, 0.25, 1.0]; 3],
                    },
                    ColorLayer {
                        name: "Corner".to_owned(),
                        domain: ColorDomain::Corner,
                        values: vec![[0.0, 0.0, 1.0, 1.0]; 3],
                    },
                ],
                active: Some("Col".to_owned()),
                render: Some("Col".to_owned()),
            },
        );
        mesh.attributes.insert(
            "weight".to_owned(),
            json!({
                "domain": "point",
                "type": "float",
                "values": {"v0": 0.5, "v1": 0.5, "v2": 0.5}
            }),
        );
        mesh.attributes.insert(
            "blender_attributes".to_owned(),
            json!([
                {"name": "Blender Col", "domain": "POINT", "data_type": "BYTE_COLOR", "values": []},
                {"name": "Pinned", "domain": "POINT", "data_type": "INT", "values": []},
                {"name": ".select_vert", "domain": "POINT", "data_type": "BOOLEAN", "values": []}
            ]),
        );
        Ok(mesh)
    }
}

#[cfg(test)]
mod tests {
    type TestResult = std::result::Result<(), Box<dyn std::error::Error>>;

    use super::{
        ExchangeVertexGroup, byte_color_roundtrip, decode_byte_color, element_data_losses,
        encode_byte_color, graph_from_meshes, mesh_attribute_losses, point_color_values,
        test_support::{triangle_with_attributes, triangle_with_layers},
        uv_layer_losses,
    };

    #[test]
    fn obj_and_ply_lose_every_layer_except_the_active_one() {
        let meshes = [triangle_with_layers()];
        for format in ["obj", "ply"] {
            let losses = uv_layer_losses(format, &meshes);
            assert_eq!(losses.len(), 1, "{format}");
            assert_eq!(losses[0].feature_id, "mesh.uv_layer");
            assert!(losses[0].reason.contains("UVMap"), "{format}");
        }
    }

    #[test]
    fn stl_loses_every_layer_and_layered_formats_lose_none() {
        let meshes = [triangle_with_layers()];
        assert_eq!(uv_layer_losses("stl", &meshes).len(), 2);
        for format in ["usda", "usdc", "fbx", "fbx-binary", "alembic", "glb"] {
            assert!(uv_layer_losses(format, &meshes).is_empty(), "{format}");
        }
    }

    #[test]
    fn alembic_loses_only_the_names_it_reserves() {
        let mut mesh = triangle_with_layers();
        mesh.uv_layers[0].name = "P".to_owned();
        let losses = uv_layer_losses("alembic", &[mesh]);
        assert_eq!(losses.len(), 1);
        assert_eq!(losses[0].feature_id, "mesh.uv_layer");
        assert!(losses[0].reason.contains("`P`"), "{}", losses[0].reason);
    }

    #[test]
    fn every_non_gltf_format_reports_color_and_generic_attributes() -> crate::error::Result<()> {
        let mesh = triangle_with_attributes()?;
        let formats = [
            "fbx",
            "fbx-binary",
            "alembic",
            "obj",
            "ply",
            "stl",
            "usda",
            "usdc",
            "usdz",
        ];
        for format in formats {
            let losses = mesh_attribute_losses(format, "triangle", &mesh);
            let colors = losses
                .iter()
                .filter(|loss| loss.feature_id == "mesh.color_attribute")
                .map(|loss| loss.reason.as_str())
                .collect::<Vec<_>>();
            let generic = losses
                .iter()
                .filter(|loss| loss.feature_id == "mesh.attribute")
                .map(|loss| loss.reason.as_str())
                .collect::<Vec<_>>();
            // The Blender color `Blender Col`, `weight`, and `Pinned`. Native color layers are
            // reported by `element_data_losses` per format, and `.select_vert` is not reported.
            assert_eq!(colors.len(), 1, "{format}: {colors:?}");
            assert_eq!(generic.len(), 2, "{format}: {generic:?}");
            assert!(
                colors[0].contains("Blender color attribute `Blender Col`"),
                "{format}"
            );
            assert!(generic[0].contains("mesh attribute `weight`"), "{format}");
            assert!(
                generic[1].contains("Blender attribute `Pinned`"),
                "{format}"
            );
            assert!(
                losses
                    .iter()
                    .all(|loss| loss.data_id.as_deref() == Some("triangle"))
            );
            assert!(
                losses
                    .iter()
                    .all(|loss| !loss.reason.contains("select_vert"))
            );
        }
        Ok(())
    }

    #[test]
    fn shading_uv_and_vertex_group_mirrors_are_not_attribute_losses() -> crate::error::Result<()> {
        let mut mesh = triangle_with_attributes()?;
        mesh.attributes.insert(
            "shade_smooth".to_owned(),
            serde_json::json!({"domain": "face", "type": "bool", "values": {}}),
        );
        mesh.attributes.insert(
            "sharp".to_owned(),
            serde_json::json!({"domain": "edge", "type": "bool", "values": {}}),
        );

        mesh.attributes.insert(
            "vertex_groups".to_owned(),
            serde_json::json!({"Top": {"v0": 0.5}}),
        );
        let losses = mesh_attribute_losses("obj", "triangle", &mesh);
        assert!(
            losses
                .iter()
                .all(|loss| !loss.reason.contains("`shade_smooth`"))
        );
        assert!(losses.iter().all(|loss| !loss.reason.contains("`sharp`")));
        assert!(
            losses
                .iter()
                .all(|loss| !loss.reason.contains("`uv_layers`"))
        );
        assert!(
            losses
                .iter()
                .all(|loss| !loss.reason.contains("`color_layers`"))
        );
        assert!(
            losses
                .iter()
                .all(|loss| !loss.reason.contains("`vertex_groups`"))
        );
        Ok(())
    }

    #[test]
    fn escaped_user_names_are_reported_by_their_user_name() -> crate::error::Result<()> {
        let mut mesh = triangle_with_attributes()?;
        mesh.attributes.insert(
            "user:sharp".to_owned(),
            serde_json::json!({"domain": "point", "type": "float", "values": {}}),
        );
        let losses = mesh_attribute_losses("obj", "triangle", &mesh);
        assert!(
            losses
                .iter()
                .any(|loss| loss.reason.ends_with("mesh attribute `sharp`"))
        );
        Ok(())
    }

    #[test]
    fn corner_colors_average_incident_corners_for_point_domain_writers() -> crate::error::Result<()>
    {
        use crate::geom::color_layers::{ColorDomain, ColorLayer};

        let mut mesh = triangle_with_layers();
        mesh.positions.push([1.0, 1.0, 0.0]);
        mesh.positions.push([2.0, 2.0, 2.0]);
        mesh.faces = vec![vec![0, 1, 2], vec![2, 1, 3]];
        let layer = ColorLayer {
            name: "paint".to_owned(),
            domain: ColorDomain::Corner,
            values: vec![
                [0.0, 0.0, 1.0, 1.0],
                [1.0, 0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0, 1.0],
                [0.0, 0.0, 1.0, 1.0],
                [0.0, 1.0, 0.0, 1.0],
                [1.0, 0.0, 0.0, 1.0],
            ],
        };
        let point = point_color_values(&mesh, &layer)?;
        assert_eq!(point[1], [0.5, 0.5, 0.0, 1.0]);
        assert_eq!(point[4], [0.0; 4]);
        Ok(())
    }

    #[test]
    fn imported_exchange_mesh_persists_colors_and_sparse_vertex_groups() -> TestResult {
        use crate::{
            geom::color_layers::{self, ColorDomain, ColorLayer, ColorStore},
            model::Id,
        };

        let mut exchange = triangle_with_layers();
        let colors = ColorStore {
            layers: vec![
                ColorLayer {
                    name: "PointPaint".to_owned(),
                    domain: ColorDomain::Point,
                    values: vec![[0.2, 0.3, 0.4, 0.5]; 3],
                },
                ColorLayer {
                    name: "CornerPaint".to_owned(),
                    domain: ColorDomain::Corner,
                    values: vec![[0.7, 0.6, 0.5, 0.4]; 3],
                },
            ],
            active: Some("PointPaint".to_owned()),
            render: Some("CornerPaint".to_owned()),
        };
        exchange.colors = colors.clone();
        exchange.vertex_groups = vec![ExchangeVertexGroup {
            name: "Weights".to_owned(),
            weights: vec![(0, 0.0), (2, 0.75)],
        }];

        let doc = graph_from_meshes(
            vec![exchange],
            "00000000-0000-4000-8000-000000000000".to_owned(),
        )?;
        let root = doc
            .collections
            .get(&Id::from_static("collection_root"))
            .ok_or("root collection missing")?;
        let node_id = root.objects.first().ok_or("imported node missing")?;
        let node = doc.nodes.get(node_id).ok_or("imported node missing")?;
        let data_id = node.data.as_ref().ok_or("imported mesh data missing")?;
        let data = doc
            .data_blocks
            .get(data_id)
            .ok_or("imported mesh data missing")?;
        let mesh = data.mesh.as_ref().ok_or("imported mesh missing")?;
        assert_eq!(color_layers::read(mesh), colors);
        assert_eq!(data.vertex_groups.len(), 1);
        assert_eq!(data.vertex_groups[0].name, "Weights");
        let group_id = &data.vertex_groups[0].id;
        assert_eq!(data.vertex_weights[&mesh.vertices[0].id][group_id], 0.0);
        assert_eq!(data.vertex_weights[&mesh.vertices[2].id][group_id], 0.75);
        assert!(!data.vertex_weights.contains_key(&mesh.vertices[1].id));
        Ok(())
    }

    #[test]
    fn evaluated_mesh_export_reads_mirror_propagated_vertex_weights() -> TestResult {
        use std::collections::BTreeMap;

        use crate::{
            eval::{EvaluationContext, Snapshot},
            geom::Mesh,
            model::{DataBlock, Id, Modifier, ModifierRuntime, Node, SceneDoc, VertexGroup},
        };

        let group_id = Id::new("weighted_group")?;
        let data_id = Id::new("weighted_mesh")?;
        let node_id = Id::new("mirrored_mesh")?;
        let mesh = Mesh::from_positions_and_faces(
            vec![
                glam::DVec3::new(1.0, 0.0, 0.0),
                glam::DVec3::new(2.0, 0.0, 0.0),
                glam::DVec3::new(1.0, 1.0, 0.0),
            ],
            vec![vec![0, 1, 2]],
        )
        .map_err(|error| {
            crate::error::PotError::new(crate::error::ErrorCode::InternalError, error.to_string())
        })?;
        let mut vertex_weights = BTreeMap::new();
        for (vertex, weight) in mesh.vertices.iter().zip([0.0, 0.5, 1.0]) {
            vertex_weights.insert(vertex.id, BTreeMap::from([(group_id.clone(), weight)]));
        }
        let modifier_params = serde_json::json!({
            "use_axis": [true, false, false],
            "use_mirror_merge": false,
            "merge_threshold": 0.001
        })
        .as_object()
        .cloned()
        .ok_or("Mirror parameters are not an object")?;
        let mut doc = SceneDoc::new("00000000-0000-4000-8000-000000000000".to_owned());
        doc.data_blocks.insert(
            data_id.clone(),
            DataBlock {
                data_type: "mesh".to_owned(),
                mesh: Some(mesh),
                vertex_groups: vec![VertexGroup {
                    id: group_id,
                    name: "Weights".to_owned(),
                }],
                vertex_weights,
                ..DataBlock::default()
            },
        );
        doc.nodes.insert(
            node_id.clone(),
            Node {
                name: "Mirror".to_owned(),
                kind: "mesh".to_owned(),
                data: Some(data_id),
                modifiers: vec![Modifier {
                    baked_ocean_cache: None,
                    id: Id::new("mirror_modifier")?,
                    modifier_type: "mirror".to_owned(),
                    name: "Mirror".to_owned(),
                    enabled: true,
                    show_viewport: true,
                    show_render: true,
                    params: modifier_params,
                    binding_data: None,
                    runtime: ModifierRuntime::default(),
                }],
                ..Node::default()
            },
        );
        doc.collections
            .get_mut(&Id::from_static("collection_root"))
            .ok_or("root collection missing")?
            .objects
            .push(node_id);
        let snapshot = Snapshot::evaluate(&doc, &EvaluationContext::default())?;
        let meshes = super::evaluated_meshes(&doc, &snapshot)?;
        let weights = &meshes[0].vertex_groups[0].weights;
        assert_eq!(meshes[0].positions.len(), 6);
        assert_eq!(weights.len(), 6);
        for expected in [0.0, 0.5, 1.0] {
            assert_eq!(
                weights.iter().filter(|entry| entry.1 == expected).count(),
                2
            );
        }
        Ok(())
    }

    #[test]
    fn blender_byte_color_roundtrip_encodes_rgb_and_alpha_differently() {
        assert!((encode_byte_color(0.5, false) - 188.0 / 255.0).abs() < 1.0e-12);
        assert!((encode_byte_color(0.5, true) - 128.0 / 255.0).abs() < 1.0e-12);
        assert!(
            (decode_byte_color(128.0 / 255.0, false) - 0.215_860_500_113_899_26).abs() < 1.0e-12
        );
        assert_eq!(decode_byte_color(128.0 / 255.0, true), 128.0 / 255.0);
        assert!((byte_color_roundtrip(0.5, false) - 0.502_886_458_032_568_7).abs() < 1.0e-9);
    }

    #[test]
    fn byte_color_formats_report_quantization_clipping_and_point_to_corner_losses() {
        use crate::geom::color_layers::{ColorDomain, ColorLayer, ColorStore};

        let mut mesh = triangle_with_layers();
        mesh.colors = ColorStore {
            layers: vec![ColorLayer {
                name: "Paint".to_owned(),
                domain: ColorDomain::Point,
                values: vec![[-0.25, 0.5, 0.25, 0.5]; 3],
            }],
            active: Some("Paint".to_owned()),
            render: Some("Paint".to_owned()),
        };
        let fbx = element_data_losses("fbx", &[mesh.clone()]);
        assert!(fbx.iter().any(|loss| loss.reason.contains("quantized")));
        assert!(fbx.iter().any(|loss| loss.reason.contains("clips")));
        assert!(fbx.iter().any(|loss| loss.reason.contains("as CORNER")));
        let alembic = element_data_losses("alembic", &[mesh]);
        assert!(alembic.iter().any(|loss| loss.reason.contains("as CORNER")));
        assert!(alembic.iter().any(|loss| loss.reason.contains("quantized")));
        assert!(alembic.iter().any(|loss| loss.reason.contains("clips")));
    }

    #[test]
    fn obj_validation_reports_overlapping_whole_face_groups() {
        let mut mesh = triangle_with_layers();
        mesh.vertex_groups = ["First", "Second"]
            .into_iter()
            .map(|name| ExchangeVertexGroup {
                name: name.to_owned(),
                weights: vec![(0, 1.0), (1, 1.0), (2, 1.0)],
            })
            .collect();
        let losses = element_data_losses("obj", &[mesh]);
        assert_eq!(
            losses
                .iter()
                .filter(|loss| loss.reason.contains("overlapping membership"))
                .count(),
            1
        );
        assert!(losses.iter().any(|loss| loss.reason.contains("`Second`")));
    }
}
