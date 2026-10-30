use std::collections::{BTreeMap, HashMap, HashSet};

use glam::{DVec3, Vec3};
use num_traits::ToPrimitive;
use serde_json::{Value, json};

use crate::{
    error::{ErrorCode, PotError, Result},
    geom::{
        Mesh, edge_key,
        provenance::{Origin, remap_attributes_with_edges},
        triangulate::{NgonMethod, Polygon, QuadMethod, triangulate_face_for_geometry_nodes},
    },
};

use super::{Field, domain_element_count, evaluation, field_bool};

pub(super) struct ExtrudeResult {
    pub(super) mesh: Mesh,
    pub(super) top_domain: &'static str,
    pub(super) top: Vec<Value>,
    pub(super) side_domain: &'static str,
    pub(super) side: Vec<Value>,
}
#[path = "mesh_dual.rs"]
mod mesh_dual;
#[path = "mesh_extrude_individual.rs"]
mod mesh_extrude_individual;
#[path = "mesh_extrude_region.rs"]
mod mesh_extrude_region;
pub(super) fn reset_untransferred_string_attributes(
    source: &Mesh,
    output: &mut Mesh,
) -> Result<()> {
    let source_corner_count = source
        .faces
        .iter()
        .try_fold(0_usize, |count, face| {
            count.checked_add(face.vertices.len())
        })
        .ok_or_else(|| evaluation("mesh corner count exceeds the supported range"))?;
    mesh_extrude_region::reset_untransferred_string_attributes(source, output, source_corner_count);
    Ok(())
}
pub(super) fn validate_extrusion_edge_attribute_types(source: &Mesh) -> Result<()> {
    mesh_extrude_region::validate_edge_attribute_types(source)
}

pub(super) fn dual_mesh(source: &Mesh, keep_boundaries: bool) -> Result<Mesh> {
    mesh_dual::evaluate(source, keep_boundaries)
}

pub(super) fn selection_mask(field: &Field, mesh: &Mesh, domain: &str) -> Result<Vec<bool>> {
    (0..domain_element_count(mesh, domain)?)
        .map(|index| field_bool(field, mesh, index, domain, false))
        .collect()
}
pub(super) fn evaluate_node(
    evaluator: &mut super::Evaluator<'_>,
    id: &super::Id,
    node: &super::GraphNode,
    output: &str,
) -> Result<super::EvalValue> {
    match node.node_type.as_str() {
        "GeometryNodeExtrudeMesh" => evaluate_extrude(evaluator, id, node, output),
        "GeometryNodeSplitEdges" => evaluate_split_edges(evaluator, id, node),
        "GeometryNodeFlipFaces" => evaluate_flip_faces(evaluator, id, node),
        "GeometryNodeMergeByDistance" => evaluate_merge_by_distance(evaluator, id, node),
        "GeometryNodeDeleteGeometry" => evaluate_delete_geometry(evaluator, id, node),
        "GeometryNodeSeparateGeometry" => evaluate_separate_geometry(evaluator, id, node, output),
        "GeometryNodeTriangulate" => evaluate_triangulate(evaluator, id, node),
        "GeometryNodeSubdivideMesh" => evaluate_subdivide(evaluator, id, node),
        "GeometryNodeScaleElements" => evaluate_scale_elements(evaluator, id, node),
        "GeometryNodeDualMesh" => evaluate_dual_mesh(evaluator, id, node),
        _ => Err(evaluation("unknown mesh operation node")),
    }
}

fn input_field(
    evaluator: &mut super::Evaluator<'_>,
    id: &super::Id,
    node: &super::GraphNode,
    socket: &str,
    default: super::EvalValue,
) -> Result<Field> {
    Ok(super::as_field(
        evaluator.input_value_with_node(id, node, socket, default)?,
    ))
}

fn input_geometry(
    evaluator: &mut super::Evaluator<'_>,
    id: &super::Id,
    node: &super::GraphNode,
    socket: &str,
) -> Result<super::GeometryValue> {
    super::into_geometry(evaluator.geometry_input(id, node, socket)?)
}

fn evaluate_split_edges(
    evaluator: &mut super::Evaluator<'_>,
    id: &super::Id,
    node: &super::GraphNode,
) -> Result<super::EvalValue> {
    let mut geometry = input_geometry(evaluator, id, node, "Mesh")?;
    let selection = input_field(
        evaluator,
        id,
        node,
        "Selection",
        super::EvalValue::Scalar(json!(true)),
    )?;
    let selection = selection_mask(&selection, &geometry.mesh, "EDGE")?;
    geometry.mesh = split_edges(&geometry.mesh, &selection)?;
    Ok(super::EvalValue::Geometry(geometry))
}

fn evaluate_flip_faces(
    evaluator: &mut super::Evaluator<'_>,
    id: &super::Id,
    node: &super::GraphNode,
) -> Result<super::EvalValue> {
    let mut geometry = input_geometry(evaluator, id, node, "Mesh")?;
    let selection = input_field(
        evaluator,
        id,
        node,
        "Selection",
        super::EvalValue::Scalar(json!(true)),
    )?;
    let selection = selection_mask(&selection, &geometry.mesh, "FACE")?;
    flip_faces(&mut geometry.mesh, &selection)?;
    Ok(super::EvalValue::Geometry(geometry))
}

fn evaluate_merge_by_distance(
    evaluator: &mut super::Evaluator<'_>,
    id: &super::Id,
    node: &super::GraphNode,
) -> Result<super::EvalValue> {
    let mut geometry = input_geometry(evaluator, id, node, "Geometry")?;
    let selection = input_field(
        evaluator,
        id,
        node,
        "Selection",
        super::EvalValue::Scalar(json!(true)),
    )?;
    let selection = selection_mask(&selection, &geometry.mesh, "POINT")?;
    let mode = super::scalar_string_input(evaluator, id, node, "Mode", "ALL")?;
    let connected = match mode.as_str() {
        "ALL" | "All" => false,
        "CONNECTED" | "Connected" => true,
        _ => return Err(evaluation("unsupported Merge by Distance mode")),
    };
    let distance = super::scalar_number_input(evaluator, id, node, "Distance", 0.001)?.max(0.0);
    geometry.mesh = merge_by_distance(&geometry.mesh, &selection, distance, connected)?;
    Ok(super::EvalValue::Geometry(geometry))
}

fn evaluate_delete_geometry(
    evaluator: &mut super::Evaluator<'_>,
    id: &super::Id,
    node: &super::GraphNode,
) -> Result<super::EvalValue> {
    let mut geometry = input_geometry(evaluator, id, node, "Geometry")?;
    let domain = super::canonical_mesh_domain(&super::property_string(node, "domain", "POINT")?)?;
    let mode = super::property_string(node, "mode", "ALL")?;
    let selection = input_field(
        evaluator,
        id,
        node,
        "Selection",
        super::EvalValue::Scalar(json!(true)),
    )?;
    let selection = selection_mask(&selection, &geometry.mesh, domain)?;
    delete_geometry(&mut geometry.mesh, domain, &mode, &selection)?;
    Ok(super::EvalValue::Geometry(geometry))
}

fn evaluate_separate_geometry(
    evaluator: &mut super::Evaluator<'_>,
    id: &super::Id,
    node: &super::GraphNode,
    output: &str,
) -> Result<super::EvalValue> {
    let geometry = input_geometry(evaluator, id, node, "Geometry")?;
    let domain = super::canonical_mesh_domain(&super::property_string(node, "domain", "POINT")?)?;
    let selection = input_field(
        evaluator,
        id,
        node,
        "Selection",
        super::EvalValue::Scalar(json!(true)),
    )?;
    let selection = selection_mask(&selection, &geometry.mesh, domain)?;
    let keep_selected = match output {
        "Selection" => true,
        "Inverted" => false,
        _ => return Err(evaluation("unknown Separate Geometry output")),
    };
    Ok(super::EvalValue::Geometry(super::GeometryValue {
        mesh: separate_geometry(&geometry.mesh, domain, &selection, keep_selected)?,
        instance_paths: geometry.instance_paths,
        instance_attributes: geometry.instance_attributes,
    }))
}

fn evaluate_triangulate(
    evaluator: &mut super::Evaluator<'_>,
    id: &super::Id,
    node: &super::GraphNode,
) -> Result<super::EvalValue> {
    let mut geometry = input_geometry(evaluator, id, node, "Mesh")?;
    let selection = input_field(
        evaluator,
        id,
        node,
        "Selection",
        super::EvalValue::Scalar(json!(true)),
    )?;
    let selection = selection_mask(&selection, &geometry.mesh, "FACE")?;
    let quad_method =
        super::scalar_string_input(evaluator, id, node, "Quad Method", "SHORTEST_DIAGONAL")?;
    let ngon_method = super::scalar_string_input(evaluator, id, node, "N-gon Method", "BEAUTY")?;
    geometry.mesh = triangulate(
        &geometry.mesh,
        &selection,
        super::parse_quad_method(&quad_method)?,
        super::parse_ngon_method(&ngon_method)?,
    )?;
    Ok(super::EvalValue::Geometry(geometry))
}

fn evaluate_scale_elements(
    evaluator: &mut super::Evaluator<'_>,
    id: &super::Id,
    node: &super::GraphNode,
) -> Result<super::EvalValue> {
    let mut geometry = input_geometry(evaluator, id, node, "Geometry")?;
    let domain = super::canonical_mesh_domain(&super::property_string(node, "domain", "FACE")?)?;
    let selection_field = input_field(
        evaluator,
        id,
        node,
        "Selection",
        super::EvalValue::Scalar(json!(true)),
    )?;
    let selection = selection_mask(&selection_field, &geometry.mesh, domain)?;
    if !selection.iter().any(|selected| *selected) {
        return Ok(super::EvalValue::Geometry(geometry));
    }
    let scale_field = input_field(
        evaluator,
        id,
        node,
        "Scale",
        super::EvalValue::Scalar(json!(1.0)),
    )?;
    let mut scales = Vec::with_capacity(selection.len());
    for index in 0..selection.len() {
        let scale = super::field_json(&scale_field, &geometry.mesh, index, domain)?
            .as_f64()
            .ok_or_else(|| evaluation("Scale Elements scale is not numeric"))?
            .to_f32()
            .ok_or_else(|| evaluation("Scale Elements scale is outside Blender's float range"))?;
        scales.push(scale);
    }
    let center_field = input_field(
        evaluator,
        id,
        node,
        "Center",
        super::EvalValue::Field(Field::Position),
    )?;
    let centers = (0..selection.len())
        .map(|index| {
            super::field_vec3_in_domain(&center_field, &geometry.mesh, index, domain, DVec3::ZERO)
                .map(DVec3::as_vec3)
        })
        .collect::<Result<Vec<_>>>()?;
    let scale_mode = super::scalar_string_input(evaluator, id, node, "Scale Mode", "UNIFORM")?;
    let single_axis = matches!(scale_mode.as_str(), "SINGLE_AXIS" | "Single Axis");
    let axes = if single_axis {
        let axis_field = input_field(
            evaluator,
            id,
            node,
            "Axis",
            super::EvalValue::Scalar(json!([1.0, 0.0, 0.0])),
        )?;
        (0..selection.len())
            .map(|index| {
                super::field_vec3_in_domain(&axis_field, &geometry.mesh, index, domain, DVec3::ZERO)
                    .map(DVec3::as_vec3)
            })
            .collect::<Result<Vec<_>>>()?
    } else {
        vec![Vec3::ZERO; selection.len()]
    };
    scale_elements(
        &mut geometry.mesh,
        domain,
        &selection,
        &scales,
        &centers,
        &axes,
        single_axis,
    )?;
    Ok(super::EvalValue::Geometry(geometry))
}

fn evaluate_subdivide(
    evaluator: &mut super::Evaluator<'_>,
    id: &super::Id,
    node: &super::GraphNode,
) -> Result<super::EvalValue> {
    let mut geometry = input_geometry(evaluator, id, node, "Mesh")?;
    let level = super::scalar_integer_input(evaluator, id, node, "Level", 1)?;
    let level = usize::try_from(level.clamp(0, 6))
        .map_err(|_| evaluation("Subdivide Mesh level is outside Blender's input range"))?;
    geometry.mesh = subdivide(&geometry.mesh, level)?;
    Ok(super::EvalValue::Geometry(geometry))
}

fn evaluate_dual_mesh(
    evaluator: &mut super::Evaluator<'_>,
    id: &super::Id,
    node: &super::GraphNode,
) -> Result<super::EvalValue> {
    let mut geometry = input_geometry(evaluator, id, node, "Mesh")?;
    let keep_boundaries = super::scalar_bool_input(evaluator, id, node, "Keep Boundaries", false)?;
    geometry.mesh = dual_mesh(&geometry.mesh, keep_boundaries)?;
    Ok(super::EvalValue::Geometry(geometry))
}

fn evaluate_extrude(
    evaluator: &mut super::Evaluator<'_>,
    id: &super::Id,
    node: &super::GraphNode,
    output: &str,
) -> Result<super::EvalValue> {
    let geometry = input_geometry(evaluator, id, node, "Mesh")?;
    let mode = super::property_string(node, "mode", "FACES")?;
    let domain = extrusion_domain(&mode)?;
    let selection_field = input_field(
        evaluator,
        id,
        node,
        "Selection",
        super::EvalValue::Scalar(json!(true)),
    )?;
    let selection = selection_mask(&selection_field, &geometry.mesh, domain)?;
    let (offsets, offset_is_single) =
        effective_extrusion_offsets(evaluator, id, node, &geometry.mesh, domain, &selection)?;
    let extruded = extrude_by_mode(
        evaluator,
        id,
        node,
        &geometry.mesh,
        &selection,
        &offsets,
        offset_is_single,
        &mode,
    )?;
    let top_base = evaluator.anonymous_attribute_name(id, "Top");
    let side_base = evaluator.anonymous_attribute_name(id, "Side");
    let top_attribute = super::unique_anonymous_attribute_name(&geometry.mesh, &top_base)?;
    let side_attribute = super::unique_anonymous_attribute_name(&geometry.mesh, &side_base)?;
    let top_is_demanded = evaluator.output_is_demanded(id, "Top");
    let side_is_demanded = evaluator.output_is_demanded(id, "Side");
    match output {
        "Mesh" => {
            let mut mesh = extruded.mesh;
            if top_is_demanded {
                evaluator.store_owned_anonymous_attribute(
                    &mut mesh,
                    &top_attribute,
                    extruded.top_domain,
                    "BOOLEAN",
                    &extruded.top,
                )?;
            }
            if side_is_demanded {
                evaluator.store_owned_anonymous_attribute(
                    &mut mesh,
                    &side_attribute,
                    extruded.side_domain,
                    "BOOLEAN",
                    &extruded.side,
                )?;
            }
            Ok(super::EvalValue::Geometry(super::GeometryValue {
                mesh,
                instance_paths: geometry.instance_paths,
                instance_attributes: geometry.instance_attributes,
            }))
        }
        "Top" => Ok(super::EvalValue::Field(Field::AnonymousAttribute {
            name: top_attribute,
            data_type: "BOOLEAN".to_owned(),
            cache: std::rc::Rc::default(),
        })),
        "Side" => Ok(super::EvalValue::Field(Field::AnonymousAttribute {
            name: side_attribute,
            data_type: "BOOLEAN".to_owned(),
            cache: std::rc::Rc::default(),
        })),
        _ => Err(evaluation("unknown Extrude Mesh output")),
    }
}

fn extrusion_domain(mode: &str) -> Result<&'static str> {
    match mode {
        "FACES" => Ok("FACE"),
        "EDGES" => Ok("EDGE"),
        "VERTICES" => Ok("POINT"),
        _ => Err(PotError::with_details(
            ErrorCode::UnsupportedFeature,
            format!("native Geometry Nodes extrusion mode `{mode}` is not implemented"),
            json!({"feature_id":format!("graph.node.GeometryNodeExtrudeMesh.mode.{mode}")}),
        )),
    }
}

fn effective_extrusion_offsets(
    evaluator: &mut super::Evaluator<'_>,
    id: &super::Id,
    node: &super::GraphNode,
    mesh: &Mesh,
    domain: &str,
    selection: &[bool],
) -> Result<(Vec<Vec3>, bool)> {
    let offset = input_field(
        evaluator,
        id,
        node,
        "Offset",
        super::EvalValue::Field(Field::Normal),
    )?;
    let offset_scale = input_field(
        evaluator,
        id,
        node,
        "Offset Scale",
        super::EvalValue::Scalar(json!(1.0)),
    )?;
    let offset_is_single = super::field_is_single(&offset) && super::field_is_single(&offset_scale);
    let vertex_normals = if matches!(&offset, Field::Normal) && matches!(domain, "EDGE" | "POINT") {
        let positions = mesh.vertices.iter().map(|vertex| vertex.co).collect::<Vec<_>>();
        Some(crate::geom::normals::geometric_vertex_normals(
            mesh, &positions,
        )?)
    } else {
        None
    };
    let vertex_indices = (domain == "EDGE" && vertex_normals.is_some()).then(|| {
        mesh.vertices
            .iter()
            .enumerate()
            .map(|(index, vertex)| (vertex.id, index))
            .collect::<HashMap<_, _>>()
    });
    let mut offsets = vec![Vec3::ZERO; selection.len()];
    let mut face_normal_calculator: Option<super::mesh_topology::FaceNormalCalculator> = None;
    for (index, selected) in selection.iter().enumerate() {
        if !selected {
            continue;
        }
        let vector = if domain == "FACE" && matches!(&offset, Field::Normal) {
            let face = mesh
                .faces
                .get(index)
                .ok_or_else(|| evaluation("extrude face index is out of range"))?;
            let normal = if let Some(calculator) = face_normal_calculator.as_ref() {
                calculator.face(face)?
            } else {
                let calculator = super::mesh_topology::FaceNormalCalculator::new(mesh);
                let normal = calculator.face(face)?;
                face_normal_calculator = Some(calculator);
                normal
            };
            DVec3::new(
                f64::from(normal.x),
                f64::from(normal.y),
                f64::from(normal.z),
            )
        } else if let Some(vertex_normals) = vertex_normals.as_ref() {
            match domain {
                "POINT" | "POINTS" => vertex_normals
                    .get(index)
                    .copied()
                    .ok_or_else(|| evaluation("extrude point normal index is out of range"))?,
                "EDGE" => edge_normal_from_vertex_normals(
                    mesh,
                    vertex_normals,
                    vertex_indices
                        .as_ref()
                        .ok_or_else(|| evaluation("edge normal vertex index map is missing"))?,
                    index,
                )?,
                _ => super::field_vec3_in_domain(&offset, mesh, index, domain, DVec3::ZERO)?,
            }
        } else {
            super::field_vec3_in_domain(&offset, mesh, index, domain, DVec3::ZERO)?
        };
        let scale = super::field_json(&offset_scale, mesh, index, domain)?
            .as_f64()
            .ok_or_else(|| evaluation("extrude offset scale is not numeric"))?
            .to_f32()
            .ok_or_else(|| evaluation("extrude offset scale is outside Blender's float range"))?;
        offsets[index] = vector.as_vec3() * scale;
    }
    Ok((offsets, offset_is_single))
}

/// Blender's NormalFieldInput edge branch: normalize the midpoint of its two vertex normals.
fn edge_normal_from_vertex_normals(
    mesh: &Mesh,
    vertex_normals: &[DVec3],
    vertex_indices: &HashMap<u32, usize>,
    edge_index: usize,
) -> Result<DVec3> {
    let edge = mesh
        .edges
        .get(edge_index)
        .ok_or_else(|| evaluation("edge normal index is out of range"))?;
    let vertex_normal = |vertex_id| {
        vertex_indices
            .get(&vertex_id)
            .and_then(|index| vertex_normals.get(*index))
            .copied()
            .map(DVec3::as_vec3)
            .ok_or_else(|| evaluation("edge normal references a missing vertex"))
    };
    let first = vertex_normal(edge.vertices[0])?;
    let second = vertex_normal(edge.vertices[1])?;
    let midpoint = Vec3::new(
        first.x * 0.5 + second.x * 0.5,
        first.y * 0.5 + second.y * 0.5,
        first.z * 0.5 + second.z * 0.5,
    );
    let length_squared = (midpoint.x * midpoint.x + midpoint.y * midpoint.y)
        + midpoint.z * midpoint.z;
    let normal = if length_squared > 1.0e-35_f32 {
        let length = length_squared.sqrt();
        Vec3::new(midpoint.x / length, midpoint.y / length, midpoint.z / length)
    } else {
        Vec3::ZERO
    };
    Ok(DVec3::new(
        f64::from(normal.x),
        f64::from(normal.y),
        f64::from(normal.z),
    ))
}

fn extrude_by_mode(
    evaluator: &mut super::Evaluator<'_>,
    id: &super::Id,
    node: &super::GraphNode,
    mesh: &Mesh,
    selection: &[bool],
    offsets: &[Vec3],
    offset_is_single: bool,
    mode: &str,
) -> Result<ExtrudeResult> {
    match mode {
        "FACES" => {
            if super::scalar_bool_input(evaluator, id, node, "Individual", true)? {
                mesh_extrude_individual::extrude_faces_individual(mesh, selection, offsets)
            } else {
                mesh_extrude_region::extrude_faces_region(
                    mesh,
                    selection,
                    offsets,
                    offset_is_single,
                )
            }
        }
        "EDGES" => {
            let extruded = super::mesh_extrude_nonface::extrude_edges(
                mesh,
                selection,
                offsets,
                offset_is_single,
            )?;
            Ok(ExtrudeResult {
                mesh: extruded.mesh,
                top_domain: extruded.top_domain,
                top: extruded.top,
                side_domain: extruded.side_domain,
                side: extruded.side,
            })
        }
        "VERTICES" => {
            let extruded = super::mesh_extrude_nonface::extrude_vertices(
                mesh,
                selection,
                offsets,
                offset_is_single,
            )?;
            Ok(ExtrudeResult {
                mesh: extruded.mesh,
                top_domain: extruded.top_domain,
                top: extruded.top,
                side_domain: extruded.side_domain,
                side: extruded.side,
            })
        }
        _ => Err(evaluation("unsupported Geometry Nodes extrusion mode")),
    }
}

pub(super) fn flip_faces(mesh: &mut Mesh, selection: &[bool]) -> Result<()> {
    if selection.len() != mesh.faces.len() {
        return Err(evaluation("face selection length does not match the mesh"));
    }
    let ids = mesh
        .faces
        .iter()
        .zip(selection)
        .filter(|(_, selected)| **selected)
        .map(|(face, _)| format!("f{}", face.id))
        .collect::<Vec<_>>();
    if ids.is_empty() {
        return Ok(());
    }
    crate::geom::edit::apply(
        mesh,
        "flip_faces",
        &json!({"elements":{"domain":"face","ids":ids}}),
    )?;
    Ok(())
}

pub(super) fn delete_geometry(
    mesh: &mut Mesh,
    domain: &str,
    mode: &str,
    selection: &[bool],
) -> Result<()> {
    let keep_selection = selection
        .iter()
        .map(|selected| !selected)
        .collect::<Vec<_>>();
    *mesh = copy_mesh_selection(mesh, domain, mode, &keep_selection)?;
    Ok(())
}

pub(super) fn separate_geometry(
    mesh: &Mesh,
    domain: &str,
    selection: &[bool],
    keep_selected: bool,
) -> Result<Mesh> {
    let selected = if keep_selected {
        selection.to_vec()
    } else {
        selection.iter().map(|selected| !selected).collect()
    };
    copy_mesh_selection(mesh, domain, "ALL", &selected)
}

fn copy_mesh_selection(
    source: &Mesh,
    domain: &str,
    mode: &str,
    selection: &[bool],
) -> Result<Mesh> {
    if selection.len() != domain_element_count(source, domain)? {
        return Err(evaluation(
            "mesh selection length does not match its domain",
        ));
    }
    let vertex_indices = source
        .vertices
        .iter()
        .enumerate()
        .map(|(index, vertex)| (vertex.id, index))
        .collect::<HashMap<_, _>>();
    let edge_indices = source
        .edges
        .iter()
        .enumerate()
        .map(|(index, edge)| (edge_key(edge.vertices[0], edge.vertices[1]), index))
        .collect::<HashMap<_, _>>();
    let face_edges = mesh_face_edges(source, &edge_indices)?;
    let masks = match mode {
        "ALL" => selected_components_all(source, domain, selection, &vertex_indices, &face_edges)?,
        "EDGE_FACE" => selected_components_keep_vertices(
            source,
            domain,
            selection,
            &vertex_indices,
            &face_edges,
        )?,
        "ONLY_FACE" => {
            selected_components_keep_edges(source, domain, selection, &vertex_indices, &face_edges)?
        }
        _ => return Err(evaluation("unsupported mesh selection mode")),
    };
    copy_mesh_components(source, &masks)
}

struct MeshSelectionMasks {
    vertices: Vec<bool>,
    edges: Vec<bool>,
    faces: Vec<bool>,
}

fn selected_components_all(
    source: &Mesh,
    domain: &str,
    selection: &[bool],
    vertex_indices: &HashMap<u32, usize>,
    face_edges: &[Vec<usize>],
) -> Result<MeshSelectionMasks> {
    let mut masks = MeshSelectionMasks {
        vertices: vec![false; source.vertices.len()],
        edges: vec![false; source.edges.len()],
        faces: vec![false; source.faces.len()],
    };
    match domain {
        "POINT" => {
            masks.vertices.clone_from_slice(selection);
            for (edge_index, edge) in source.edges.iter().enumerate() {
                masks.edges[edge_index] = masks.vertices[vertex_indices[&edge.vertices[0]]]
                    && masks.vertices[vertex_indices[&edge.vertices[1]]];
            }
            for (face_index, face) in source.faces.iter().enumerate() {
                masks.faces[face_index] = face
                    .vertices
                    .iter()
                    .all(|vertex| masks.vertices[vertex_indices[vertex]]);
            }
        }
        "EDGE" => {
            masks.edges.clone_from_slice(selection);
            for (edge_index, edge) in source.edges.iter().enumerate() {
                if masks.edges[edge_index] {
                    masks.vertices[vertex_indices[&edge.vertices[0]]] = true;
                    masks.vertices[vertex_indices[&edge.vertices[1]]] = true;
                }
            }
            for (face_index, edges) in face_edges.iter().enumerate() {
                masks.faces[face_index] = edges.iter().all(|edge| masks.edges[*edge]);
            }
        }
        "FACE" => {
            masks.faces.clone_from_slice(selection);
            for (face_index, face) in source.faces.iter().enumerate() {
                if masks.faces[face_index] {
                    for vertex in &face.vertices {
                        masks.vertices[vertex_indices[vertex]] = true;
                    }
                    for edge in &face_edges[face_index] {
                        masks.edges[*edge] = true;
                    }
                }
            }
        }
        _ => return Err(evaluation("unsupported mesh selection domain")),
    }
    Ok(masks)
}

fn selected_components_keep_vertices(
    source: &Mesh,
    domain: &str,
    selection: &[bool],
    vertex_indices: &HashMap<u32, usize>,
    face_edges: &[Vec<usize>],
) -> Result<MeshSelectionMasks> {
    let mut masks = MeshSelectionMasks {
        vertices: vec![true; source.vertices.len()],
        edges: vec![false; source.edges.len()],
        faces: vec![false; source.faces.len()],
    };
    match domain {
        "POINT" => {
            for (edge_index, edge) in source.edges.iter().enumerate() {
                masks.edges[edge_index] = selection[vertex_indices[&edge.vertices[0]]]
                    && selection[vertex_indices[&edge.vertices[1]]];
            }
            for (face_index, face) in source.faces.iter().enumerate() {
                masks.faces[face_index] = face
                    .vertices
                    .iter()
                    .all(|vertex| selection[vertex_indices[vertex]]);
            }
        }
        "EDGE" => {
            masks.edges.clone_from_slice(selection);
            for (face_index, edges) in face_edges.iter().enumerate() {
                masks.faces[face_index] = edges.iter().all(|edge| selection[*edge]);
            }
        }
        "FACE" => {
            masks.faces.clone_from_slice(selection);
            for (face_index, edges) in face_edges.iter().enumerate() {
                if masks.faces[face_index] {
                    for edge in edges {
                        masks.edges[*edge] = true;
                    }
                }
            }
        }
        _ => return Err(evaluation("unsupported mesh selection domain")),
    }
    Ok(masks)
}

fn selected_components_keep_edges(
    source: &Mesh,
    domain: &str,
    selection: &[bool],
    vertex_indices: &HashMap<u32, usize>,
    face_edges: &[Vec<usize>],
) -> Result<MeshSelectionMasks> {
    let mut masks = MeshSelectionMasks {
        vertices: vec![true; source.vertices.len()],
        edges: vec![true; source.edges.len()],
        faces: vec![false; source.faces.len()],
    };
    match domain {
        "POINT" => {
            for (face_index, face) in source.faces.iter().enumerate() {
                masks.faces[face_index] = face
                    .vertices
                    .iter()
                    .all(|vertex| selection[vertex_indices[vertex]]);
            }
        }
        "EDGE" => {
            for (face_index, edges) in face_edges.iter().enumerate() {
                masks.faces[face_index] = edges.iter().all(|edge| selection[*edge]);
            }
        }
        "FACE" => masks.faces.clone_from_slice(selection),
        _ => return Err(evaluation("unsupported mesh selection domain")),
    }
    Ok(masks)
}

fn mesh_face_edges(
    mesh: &Mesh,
    edge_indices: &HashMap<(u32, u32), usize>,
) -> Result<Vec<Vec<usize>>> {
    mesh.faces
        .iter()
        .map(|face| {
            (0..face.vertices.len())
                .map(|corner| {
                    edge_indices
                        .get(&edge_key(
                            face.vertices[corner],
                            face.vertices[(corner + 1) % face.vertices.len()],
                        ))
                        .copied()
                        .ok_or_else(|| evaluation("mesh face boundary edge is missing"))
                })
                .collect()
        })
        .collect()
}

fn copy_mesh_components(source: &Mesh, masks: &MeshSelectionMasks) -> Result<Mesh> {
    if masks.vertices.iter().all(|selected| *selected)
        && masks.edges.iter().all(|selected| *selected)
        && masks.faces.iter().all(|selected| *selected)
    {
        return Ok(source.clone());
    }
    let mut output = Mesh::default();
    let mut vertex_ids = HashMap::<u32, u32>::new();
    let mut vertex_origins = Vec::new();
    for (index, vertex) in source.vertices.iter().enumerate() {
        if masks.vertices[index] {
            let id = output.insert_vertex(vertex.co).map_err(mesh_error)?;
            vertex_ids.insert(vertex.id, id);
            vertex_origins.push(vec![(index, 1.0)]);
        }
    }
    if output.vertices.is_empty() {
        return Ok(output);
    }
    let mut edge_origins = Vec::new();
    for (index, edge) in source.edges.iter().enumerate() {
        if !masks.edges[index] {
            continue;
        }
        let first = *vertex_ids
            .get(&edge.vertices[0])
            .ok_or_else(|| evaluation("selected edge has an unselected endpoint"))?;
        let second = *vertex_ids
            .get(&edge.vertices[1])
            .ok_or_else(|| evaluation("selected edge has an unselected endpoint"))?;
        output.insert_edge([first, second]).map_err(mesh_error)?;
        edge_origins.push(vec![(index, 1.0)]);
    }
    let mut face_origins = Vec::new();
    let mut corner_origins = Vec::new();
    let mut corner_offsets = Vec::with_capacity(source.faces.len());
    let mut corner_offset = 0;
    for face in &source.faces {
        corner_offsets.push(corner_offset);
        corner_offset += face.vertices.len();
    }
    for (face_index, face) in source.faces.iter().enumerate() {
        if !masks.faces[face_index] {
            continue;
        }
        let vertices = face
            .vertices
            .iter()
            .map(|vertex| {
                vertex_ids
                    .get(vertex)
                    .copied()
                    .ok_or_else(|| evaluation("selected face has an unselected vertex"))
            })
            .collect::<Result<Vec<_>>>()?;
        output
            .insert_face(vertices, face.material_index)
            .map_err(mesh_error)?;
        face_origins.push(Some(face_index));
        corner_origins.extend(
            (0..face.vertices.len()).map(|corner| vec![(corner_offsets[face_index] + corner, 1.0)]),
        );
    }
    if edge_origins.len() != output.edges.len() {
        return Err(evaluation(
            "mesh selection output edge provenance is incomplete",
        ));
    }
    let origin = Origin {
        vertices: vertex_origins,
        faces: face_origins,
        corners: Some(corner_origins),
        ..Default::default()
    };
    remap_attributes_with_edges(source, &mut output, &origin, edge_origins)?;
    copy_direct_matrix_attributes(source, &mut output, &origin);
    Ok(output)
}

fn mesh_corner_offsets(mesh: &Mesh) -> Vec<usize> {
    let mut offsets = Vec::with_capacity(mesh.faces.len());
    let mut corner = 0;
    for face in &mesh.faces {
        offsets.push(corner);
        corner += face.vertices.len();
    }
    offsets
}

pub(super) fn split_edges(source: &Mesh, selection: &[bool]) -> Result<Mesh> {
    if selection.len() != source.edges.len() {
        return Err(evaluation(
            "edge split selection length does not match edges",
        ));
    }
    if !selection.iter().any(|selected| *selected) {
        return Ok(source.clone());
    }
    let topology = split_topology(source)?;
    let mut result = source.clone();
    let vertex_data = split_vertex_fans(source, &topology, selection, &mut result)?;
    let edge_origins =
        rebuild_split_edges(source, &topology, selection, &vertex_data, &mut result)?;
    let origin = Origin {
        vertices: vertex_data.origins,
        faces: (0..source.faces.len()).map(Some).collect(),
        corners: Some(
            (0..topology.corner_to_face.len())
                .map(|corner| vec![(corner, 1.0)])
                .collect(),
        ),
        ..Default::default()
    };
    remap_attributes_with_edges(source, &mut result, &origin, edge_origins)?;
    copy_direct_matrix_attributes(source, &mut result, &origin);
    Ok(result)
}

fn copy_direct_matrix_attributes(source: &Mesh, output: &mut Mesh, origin: &Origin) {
    let Some(source_layers) = source
        .attributes
        .get("blender_attributes")
        .and_then(serde_json::Value::as_array)
    else {
        return;
    };
    let Some(output_layers) = output
        .attributes
        .get_mut("blender_attributes")
        .and_then(serde_json::Value::as_array_mut)
    else {
        return;
    };
    let vertex_indices = output
        .vertices
        .iter()
        .enumerate()
        .map(|(index, vertex)| (vertex.id, index))
        .collect::<HashMap<_, _>>();
    let source_edges = source
        .edges
        .iter()
        .enumerate()
        .map(|(index, edge)| (edge_key(edge.vertices[0], edge.vertices[1]), index))
        .collect::<HashMap<_, _>>();
    let source_vertex_ids = source
        .vertices
        .iter()
        .map(|vertex| vertex.id)
        .collect::<Vec<_>>();
    for layer in source_layers {
        if !matches!(
            layer.get("data_type").and_then(serde_json::Value::as_str),
            Some("FLOAT4X4" | "MATRIX")
        ) {
            continue;
        }
        let Some(name) = layer.get("name").and_then(serde_json::Value::as_str) else {
            continue;
        };
        let Some(source_values) = layer.get("values").and_then(serde_json::Value::as_array) else {
            continue;
        };
        let Some(output_layer) = output_layers.iter_mut().find(|candidate| {
            candidate.get("name").and_then(serde_json::Value::as_str) == Some(name)
        }) else {
            continue;
        };
        let Some(output_values) = output_layer
            .get_mut("values")
            .and_then(serde_json::Value::as_array_mut)
        else {
            continue;
        };
        let domain = layer.get("domain").and_then(serde_json::Value::as_str);
        let source_indices = match domain {
            Some("POINT") => origin
                .vertices
                .iter()
                .map(|origins| single_origin(origins))
                .collect::<Vec<_>>(),
            Some("FACE") => origin.faces.clone(),
            Some("CORNER") => origin
                .corners
                .as_ref()
                .map(|corners| {
                    corners
                        .iter()
                        .map(|origins| single_origin(origins))
                        .collect()
                })
                .unwrap_or_default(),
            Some("EDGE") => output
                .edges
                .iter()
                .filter_map(|edge| {
                    let first = vertex_indices.get(&edge.vertices[0]).copied()?;
                    let second = vertex_indices.get(&edge.vertices[1]).copied()?;
                    let first_source = origin.vertices.get(first)?.first()?.0;
                    let second_source = origin.vertices.get(second)?.first()?.0;
                    let key = edge_key(
                        *source_vertex_ids.get(first_source)?,
                        *source_vertex_ids.get(second_source)?,
                    );
                    source_edges.get(&key).copied()
                })
                .map(Some)
                .collect(),
            _ => continue,
        };
        for (output_index, source_index) in source_indices.iter().enumerate() {
            if let (Some(source_value), Some(output_value)) = (
                source_index.and_then(|index| source_values.get(index)),
                output_values.get_mut(output_index),
            ) {
                *output_value = source_value.clone();
            }
        }
    }
}

fn single_origin(origins: &[(usize, f64)]) -> Option<usize> {
    (origins.len() == 1).then(|| origins[0].0)
}

struct SplitTopology {
    vertex_indices: HashMap<u32, usize>,
    corner_offsets: Vec<usize>,
    corner_to_face: Vec<usize>,
    corner_to_local: Vec<usize>,
    corner_to_edge: Vec<usize>,
    edge_corners: Vec<Vec<usize>>,
    vertex_corners: Vec<Vec<usize>>,
    vertex_edges: Vec<Vec<usize>>,
}

struct SplitVertexData {
    corner_vertices: Vec<u32>,
    loose_vertices: HashMap<(usize, usize), u32>,
    origins: Vec<Vec<(usize, f64)>>,
}

fn split_topology(source: &Mesh) -> Result<SplitTopology> {
    let vertex_indices = source
        .vertices
        .iter()
        .enumerate()
        .map(|(index, vertex)| (vertex.id, index))
        .collect::<HashMap<_, _>>();
    let edge_indices = source
        .edges
        .iter()
        .enumerate()
        .map(|(index, edge)| (edge_key(edge.vertices[0], edge.vertices[1]), index))
        .collect::<HashMap<_, _>>();
    let corner_count = source.faces.iter().map(|face| face.vertices.len()).sum();
    let mut corner_offsets = Vec::with_capacity(source.faces.len());
    let mut corner_to_face = vec![0; corner_count];
    let mut corner_to_local = vec![0; corner_count];
    let mut corner_to_edge = vec![0; corner_count];
    let mut edge_corners = vec![Vec::new(); source.edges.len()];
    let mut vertex_corners = vec![Vec::new(); source.vertices.len()];
    let mut corner_offset = 0;
    for (face_index, face) in source.faces.iter().enumerate() {
        corner_offsets.push(corner_offset);
        for local in 0..face.vertices.len() {
            let corner = corner_offset + local;
            let vertex_index = *vertex_indices
                .get(&face.vertices[local])
                .ok_or_else(|| evaluation("split face references a missing vertex"))?;
            let edge_index = *edge_indices
                .get(&edge_key(
                    face.vertices[local],
                    face.vertices[(local + 1) % face.vertices.len()],
                ))
                .ok_or_else(|| evaluation("split face boundary edge is missing"))?;
            corner_to_face[corner] = face_index;
            corner_to_local[corner] = local;
            corner_to_edge[corner] = edge_index;
            edge_corners[edge_index].push(corner);
            vertex_corners[vertex_index].push(corner);
        }
        corner_offset += face.vertices.len();
    }
    let mut vertex_edges = vec![Vec::new(); source.vertices.len()];
    for (edge_index, edge) in source.edges.iter().enumerate() {
        for vertex_id in edge.vertices {
            vertex_edges[vertex_indices[&vertex_id]].push(edge_index);
        }
    }
    Ok(SplitTopology {
        vertex_indices,
        corner_offsets,
        corner_to_face,
        corner_to_local,
        corner_to_edge,
        edge_corners,
        vertex_corners,
        vertex_edges,
    })
}

fn split_vertex_fans(
    source: &Mesh,
    topology: &SplitTopology,
    selection: &[bool],
    result: &mut Mesh,
) -> Result<SplitVertexData> {
    let corner_count = topology.corner_to_face.len();
    let mut origins = source
        .vertices
        .iter()
        .enumerate()
        .map(|(index, _)| vec![(index, 1.0)])
        .collect::<Vec<_>>();
    let mut corner_vertices = vec![0_u32; corner_count];
    let mut loose_vertices = HashMap::<(usize, usize), u32>::new();
    let mut affected = vec![false; source.vertices.len()];
    for (edge_index, edge) in source.edges.iter().enumerate() {
        if selection[edge_index] {
            for vertex_id in edge.vertices {
                affected[topology.vertex_indices[&vertex_id]] = true;
            }
        }
    }

    for (vertex_index, corners) in topology.vertex_corners.iter().enumerate() {
        if !affected[vertex_index] {
            for &corner in corners {
                corner_vertices[corner] = source.vertices[vertex_index].id;
            }
            continue;
        }
        let mut groups = Vec::<Vec<usize>>::new();
        let mut visited = HashSet::with_capacity(corners.len());
        for &start in corners {
            if !visited.insert(start) {
                continue;
            }
            let mut group = Vec::new();
            let mut pending = vec![start];
            while let Some(corner) = pending.pop() {
                group.push(corner);
                let face_index = topology.corner_to_face[corner];
                let face = &source.faces[face_index];
                let previous = topology.corner_offsets[face_index]
                    + (topology.corner_to_local[corner] + face.vertices.len() - 1)
                        % face.vertices.len();
                for edge_index in [
                    topology.corner_to_edge[corner],
                    topology.corner_to_edge[previous],
                ] {
                    if selection[edge_index] {
                        continue;
                    }
                    for &other in &topology.edge_corners[edge_index] {
                        if topology.corner_to_face[other] == face_index {
                            continue;
                        }
                        let other_face = &source.faces[topology.corner_to_face[other]];
                        let other_local = topology.corner_to_local[other];
                        let candidate = if other_face.vertices[other_local]
                            == source.vertices[vertex_index].id
                        {
                            other
                        } else {
                            topology.corner_offsets[topology.corner_to_face[other]]
                                + (other_local + 1) % other_face.vertices.len()
                        };
                        if visited.insert(candidate) {
                            pending.push(candidate);
                        }
                    }
                }
            }
            group.sort_unstable();
            groups.push(group);
        }
        let original = source.vertices[vertex_index].id;
        for group in groups.iter().take(groups.len().saturating_sub(1)) {
            let duplicate = result
                .insert_vertex(source.vertices[vertex_index].co)
                .map_err(mesh_error)?;
            origins.push(vec![(vertex_index, 1.0)]);
            for &corner in group {
                corner_vertices[corner] = duplicate;
            }
        }
        if let Some(group) = groups.last() {
            for &corner in group {
                corner_vertices[corner] = original;
            }
        }
        let loose_edges = topology.vertex_edges[vertex_index]
            .iter()
            .copied()
            .filter(|edge| topology.edge_corners[*edge].is_empty())
            .collect::<Vec<_>>();
        let selected_loose = loose_edges
            .iter()
            .copied()
            .filter(|edge| selection[*edge])
            .collect::<Vec<_>>();
        let has_unselected_loose = loose_edges.iter().any(|edge| !selection[*edge]);
        for (index, edge) in selected_loose.iter().copied().enumerate() {
            let reuse_original =
                groups.is_empty() && !has_unselected_loose && index + 1 == selected_loose.len();
            let duplicate = if reuse_original {
                original
            } else {
                let duplicate = result
                    .insert_vertex(source.vertices[vertex_index].co)
                    .map_err(mesh_error)?;
                origins.push(vec![(vertex_index, 1.0)]);
                duplicate
            };
            loose_vertices.insert((edge, vertex_index), duplicate);
        }
    }
    Ok(SplitVertexData {
        corner_vertices,
        loose_vertices,
        origins,
    })
}

fn rebuild_split_edges(
    source: &Mesh,
    topology: &SplitTopology,
    selection: &[bool],
    vertex_data: &SplitVertexData,
    result: &mut Mesh,
) -> Result<Vec<Vec<(usize, f64)>>> {
    for (face_index, face) in result.faces.iter_mut().enumerate() {
        for (local, vertex) in face.vertices.iter_mut().enumerate() {
            *vertex = vertex_data.corner_vertices[topology.corner_offsets[face_index] + local];
        }
    }
    let mut extra_edges = Vec::<(usize, [u32; 2])>::new();
    for (edge_index, edge) in source.edges.iter().enumerate() {
        if topology.edge_corners[edge_index].is_empty() {
            if selection[edge_index] {
                let first = topology.vertex_indices[&edge.vertices[0]];
                let second = topology.vertex_indices[&edge.vertices[1]];
                result.edges[edge_index].vertices = [
                    vertex_data.loose_vertices[&(edge_index, first)],
                    vertex_data.loose_vertices[&(edge_index, second)],
                ];
            }
            continue;
        }
        let mut pairs = Vec::<[u32; 2]>::new();
        for &corner in &topology.edge_corners[edge_index] {
            let face_index = topology.corner_to_face[corner];
            let local = topology.corner_to_local[corner];
            let pair = [
                vertex_data.corner_vertices[corner],
                vertex_data.corner_vertices[topology.corner_offsets[face_index]
                    + (local + 1) % source.faces[face_index].vertices.len()],
            ];
            let ordered = if pair[0] <= pair[1] {
                pair
            } else {
                [pair[1], pair[0]]
            };
            if !pairs.contains(&ordered) {
                pairs.push(ordered);
            }
        }
        if !selection[edge_index] {
            let first_corner = topology.edge_corners[edge_index][0];
            let face_index = topology.corner_to_face[first_corner];
            let local = topology.corner_to_local[first_corner];
            let second_corner = topology.corner_offsets[face_index]
                + (local + 1) % source.faces[face_index].vertices.len();
            let first = vertex_data.corner_vertices[first_corner];
            let second = vertex_data.corner_vertices[second_corner];
            let same_direction = source.faces[face_index].vertices[local] == edge.vertices[0];
            result.edges[edge_index].vertices = if same_direction {
                [first, second]
            } else {
                [second, first]
            };
            continue;
        }
        if let Some(first) = pairs.first() {
            result.edges[edge_index].vertices = *first;
            extra_edges.extend(pairs.into_iter().skip(1).map(|pair| (edge_index, pair)));
        }
    }
    let mut edge_origins = (0..source.edges.len())
        .map(|index| vec![(index, 1.0)])
        .collect::<Vec<_>>();
    for (source_edge, vertices) in extra_edges {
        result.insert_edge(vertices).map_err(mesh_error)?;
        edge_origins.push(vec![(source_edge, 1.0)]);
    }
    Ok(edge_origins)
}
pub(super) fn triangulate(
    source: &Mesh,
    selection: &[bool],
    quad_method: QuadMethod,
    ngon_method: NgonMethod,
) -> Result<Mesh> {
    if selection.len() != source.faces.len() {
        return Err(evaluation(
            "triangulate selection length does not match faces",
        ));
    }
    if source.faces.iter().all(|face| face.vertices.len() == 3) {
        return Ok(source.clone());
    }
    if !selection.iter().any(|selected| *selected) {
        return Ok(source.clone());
    }
    let triangles = triangulated_faces(source, quad_method, ngon_method)?;
    rebuild_triangulated_mesh(source, selection, &triangles)
}

fn triangulated_faces(
    source: &Mesh,
    quad_method: QuadMethod,
    ngon_method: NgonMethod,
) -> Result<Vec<Vec<Polygon>>> {
    let positions = source
        .vertices
        .iter()
        .map(|vertex| vertex.co.as_vec3().to_array())
        .collect::<Vec<_>>();
    let vertex_indices = source
        .vertices
        .iter()
        .enumerate()
        .map(|(index, vertex)| (vertex.id, index))
        .collect::<HashMap<_, _>>();
    let face_positions = source
        .faces
        .iter()
        .map(|face| {
            face.vertices
                .iter()
                .map(|id| {
                    vertex_indices
                        .get(id)
                        .map(|index| positions[*index])
                        .ok_or_else(|| evaluation("triangulate face references a missing vertex"))
                })
                .collect::<Result<Vec<_>>>()
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(face_positions
        .iter()
        .enumerate()
        .map(|(face, corners)| {
            triangulate_face_for_geometry_nodes(corners, quad_method, ngon_method)
                .into_iter()
                .map(|corners| Polygon {
                    face,
                    corners: corners.to_vec(),
                })
                .collect()
        })
        .collect())
}

fn rebuild_triangulated_mesh(
    source: &Mesh,
    selection: &[bool],
    triangles_by_face: &[Vec<Polygon>],
) -> Result<Mesh> {
    let vertex_indices = source
        .vertices
        .iter()
        .enumerate()
        .map(|(index, vertex)| (vertex.id, index))
        .collect::<HashMap<_, _>>();
    let mut output = Mesh::default();
    let mut vertex_ids = Vec::with_capacity(source.vertices.len());
    let mut output_vertex_sources = HashMap::with_capacity(source.vertices.len());
    for (source_index, vertex) in source.vertices.iter().enumerate() {
        let output_id = output.insert_vertex(vertex.co).map_err(mesh_error)?;
        vertex_ids.push(output_id);
        output_vertex_sources.insert(output_id, source_index);
    }
    let mut source_edges = HashMap::with_capacity(source.edges.len());
    for (index, edge) in source.edges.iter().enumerate() {
        let first = vertex_ids[vertex_indices[&edge.vertices[0]]];
        let second = vertex_ids[vertex_indices[&edge.vertices[1]]];
        output.insert_edge([first, second]).map_err(mesh_error)?;
        source_edges.insert(edge_key(edge.vertices[0], edge.vertices[1]), index);
    }
    let corner_offsets = mesh_corner_offsets(source);
    let mut output_faces = Vec::new();
    let mut face_sources = Vec::new();
    let mut corner_sources = Vec::new();
    for (face_index, face) in source.faces.iter().enumerate() {
        if selection[face_index] {
            for polygon in &triangles_by_face[face_index] {
                append_triangulated_polygon(
                    source,
                    face_index,
                    &polygon.corners,
                    &vertex_indices,
                    &vertex_ids,
                    &corner_offsets,
                    &mut output_faces,
                    &mut face_sources,
                    &mut corner_sources,
                );
            }
        } else {
            let corners = (0..face.vertices.len()).collect::<Vec<_>>();
            append_triangulated_polygon(
                source,
                face_index,
                &corners,
                &vertex_indices,
                &vertex_ids,
                &corner_offsets,
                &mut output_faces,
                &mut face_sources,
                &mut corner_sources,
            );
        }
    }
    output
        .insert_faces(output_faces.into_iter().map(Ok))
        .map_err(mesh_error)?;
    let edge_origins = output
        .edges
        .iter()
        .map(|edge| {
            let first = output_vertex_sources
                .get(&edge.vertices[0])
                .and_then(|index| source.vertices.get(*index));
            let second = output_vertex_sources
                .get(&edge.vertices[1])
                .and_then(|index| source.vertices.get(*index));
            first
                .zip(second)
                .and_then(|(first, second)| source_edges.get(&edge_key(first.id, second.id)))
                .map_or_else(Vec::new, |index| vec![(*index, 1.0)])
        })
        .collect();
    let origin = Origin {
        vertices: (0..source.vertices.len())
            .map(|index| vec![(index, 1.0)])
            .collect(),
        faces: face_sources,
        corners: Some(corner_sources),
        ..Default::default()
    };
    remap_attributes_with_edges(source, &mut output, &origin, edge_origins)?;
    copy_direct_matrix_attributes(source, &mut output, &origin);
    Ok(output)
}

fn append_triangulated_polygon(
    source: &Mesh,
    face_index: usize,
    corners: &[usize],
    vertex_indices: &HashMap<u32, usize>,
    vertex_ids: &[u32],
    corner_offsets: &[usize],
    output_faces: &mut Vec<(Vec<u32>, u32)>,
    face_sources: &mut Vec<Option<usize>>,
    corner_sources: &mut Vec<Vec<(usize, f64)>>,
) {
    let face = &source.faces[face_index];
    output_faces.push((
        corners
            .iter()
            .map(|corner| vertex_ids[vertex_indices[&face.vertices[*corner]]])
            .collect(),
        face.material_index,
    ));
    face_sources.push(Some(face_index));
    corner_sources.extend(
        corners
            .iter()
            .map(|corner| vec![(corner_offsets[face_index] + corner, 1.0)]),
    );
}

pub(super) fn subdivide(source: &Mesh, level: usize) -> Result<Mesh> {
    crate::geom::modifiers::subdivide_mesh_for_geometry_nodes(source, level)
}

pub(super) fn merge_by_distance(
    mesh: &Mesh,
    selection: &[bool],
    distance: f64,
    connected: bool,
) -> Result<Mesh> {
    crate::geom::modifiers::merge_by_distance_for_geometry_nodes(
        mesh, distance, connected, selection,
    )
}

pub(super) fn scale_elements(
    mesh: &mut Mesh,
    domain: &str,
    selection: &[bool],
    scales: &[f32],
    centers: &[Vec3],
    axes: &[Vec3],
    single_axis: bool,
) -> Result<()> {
    if !matches!(domain, "EDGE" | "FACE") {
        return Err(evaluation("Scale Elements supports EDGE and FACE domains"));
    }
    let element_count = domain_element_count(mesh, domain)?;
    if selection.len() != element_count
        || scales.len() != element_count
        || centers.len() != element_count
        || axes.len() != element_count
    {
        return Err(evaluation(
            "Scale Elements field length does not match its domain",
        ));
    }
    if !selection.iter().any(|selected| *selected) {
        return Ok(());
    }
    let mut parent: Vec<usize> = (0..element_count).collect();
    let mut first_for_vertex = HashMap::<u32, usize>::new();
    for (element_index, selected) in selection.iter().enumerate() {
        if !selected {
            continue;
        }
        for &vertex_id in element_vertices(mesh, domain, element_index)? {
            if let Some(previous) = first_for_vertex.get(&vertex_id) {
                union(&mut parent, element_index, *previous);
            } else {
                first_for_vertex.insert(vertex_id, element_index);
            }
        }
    }
    let mut groups = BTreeMap::<usize, Vec<usize>>::new();
    for (index, selected) in selection.iter().enumerate() {
        if *selected {
            let root = find_root(&mut parent, index);
            groups.entry(root).or_default().push(index);
        }
    }
    let vertex_indices = mesh
        .vertices
        .iter()
        .enumerate()
        .map(|(index, vertex)| (vertex.id, index))
        .collect::<HashMap<_, _>>();
    let mut next_positions = mesh
        .vertices
        .iter()
        .map(|vertex| vertex.co.as_vec3())
        .collect::<Vec<_>>();
    let mut group_vertices = Vec::with_capacity(mesh.vertices.len());
    let mut vertex_in_group = vec![false; mesh.vertices.len()];
    for group in groups.values() {
        let count = group
            .len()
            .to_f32()
            .ok_or_else(|| evaluation("scale island is too large"))?;
        let scale = group.iter().map(|index| scales[*index]).sum::<f32>() / count;
        let center = group.iter().map(|index| centers[*index]).sum::<Vec3>() / count;
        let axis = if single_axis {
            let axis = group.iter().map(|index| axes[*index]).sum::<Vec3>() / count;
            if axis.length_squared() == 0.0 {
                Vec3::X
            } else {
                axis.normalize()
            }
        } else {
            Vec3::ZERO
        };
        for &element in group {
            for &vertex_id in element_vertices(mesh, domain, element)? {
                let index = vertex_indices[&vertex_id];
                if !vertex_in_group[index] {
                    vertex_in_group[index] = true;
                    group_vertices.push(index);
                }
            }
        }
        for &index in &group_vertices {
            let position = next_positions[index];
            let difference = position - center;
            next_positions[index] = if single_axis {
                position + axis * (difference.dot(axis) * (scale - 1.0))
            } else {
                center + difference * scale
            };
            vertex_in_group[index] = false;
        }
        group_vertices.clear();
    }
    for (vertex, position) in mesh.vertices.iter_mut().zip(next_positions) {
        vertex.co = DVec3::from(position);
    }
    Ok(())
}

fn element_vertices<'a>(mesh: &'a Mesh, domain: &str, index: usize) -> Result<&'a [u32]> {
    match domain {
        "EDGE" => mesh
            .edges
            .get(index)
            .map(|edge| edge.vertices.as_slice())
            .ok_or_else(|| evaluation("scale edge index is out of range")),
        "FACE" => mesh
            .faces
            .get(index)
            .map(|face| face.vertices.as_slice())
            .ok_or_else(|| evaluation("scale face index is out of range")),
        _ => Err(evaluation("Scale Elements supports EDGE and FACE domains")),
    }
}

fn union(parent: &mut [usize], first: usize, second: usize) {
    let first = find_root(parent, first);
    let second = find_root(parent, second);
    if first != second {
        parent[second] = first.min(second);
        parent[first] = first.min(second);
    }
}

fn find_root(parent: &mut [usize], index: usize) -> usize {
    if parent[index] != index {
        parent[index] = find_root(parent, parent[index]);
    }
    parent[index]
}

fn mesh_error(error: crate::geom::MeshError) -> PotError {
    PotError::new(ErrorCode::EvaluationFailed, error.to_string())
}
