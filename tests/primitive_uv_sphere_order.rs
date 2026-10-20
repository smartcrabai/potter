// SPDX-License-Identifier: GPL-3.0-or-later
//
// Blender 5.2.2 `primitive_uv_sphere_add` oracle for Potter's `uv_sphere` primitive.
//
// Vertex positions and order must match exactly. Blender's edge and face order is run-dependent
// (the oracle marks it `edge_face_order: run_dependent`), so edges are compared as an undirected set
// and faces as a set of exact loop lists, which fixes each face's loop start.

use std::{
    collections::{BTreeSet, HashMap},
    error::Error,
};

use potter_core::geom::{Mesh, UvSphereParams};
use serde_json::Value;

type BoxError = Box<dyn Error>;
type TestResult = Result<(), BoxError>;

const ORACLE: &str = include_str!("fixtures/primitive_uvs/primitive_uv_oracle.json");

fn integer(kwargs: &Value, name: &str, default: u64) -> Result<u64, BoxError> {
    match kwargs.get(name) {
        None => Ok(default),
        Some(found) => found
            .as_u64()
            .ok_or_else(|| format!("{name} must be an integer").into()),
    }
}

fn radius(kwargs: &Value) -> Result<f64, BoxError> {
    match kwargs.get("radius") {
        None => Ok(1.0),
        Some(found) => found
            .as_f64()
            .ok_or_else(|| "radius must be a number".into()),
    }
}

fn oracle_indices(value: &Value) -> Result<Vec<usize>, BoxError> {
    value
        .as_array()
        .ok_or("index list must be an array")?
        .iter()
        .map(|item| {
            let raw = item.as_u64().ok_or("index must be an integer")?;
            usize::try_from(raw).map_err(BoxError::from)
        })
        .collect()
}

fn undirected(a: usize, b: usize) -> (usize, usize) {
    (a.min(b), a.max(b))
}

#[test]
fn uv_sphere_storage_order_matches_blender_primitive_uv_sphere_add() -> TestResult {
    let oracle: Value = serde_json::from_str(ORACLE)?;
    assert_eq!(oracle["blender_version"], "5.2.2");
    let cases = oracle["cases"].as_object().ok_or("oracle cases missing")?;
    let mut checked = 0;
    for (name, case) in cases.iter().filter(|(_, case)| case["kind"] == "uv_sphere") {
        assert_eq!(
            case["edge_face_order"], "run_dependent",
            "{name} must record that its edge and face order is run-dependent"
        );
        let kwargs = &case["kwargs"];
        let params = UvSphereParams {
            segments: u32::try_from(integer(kwargs, "segments", 32)?)?,
            ring_count: u32::try_from(integer(kwargs, "ring_count", 16)?)?,
            radius: radius(kwargs)?,
        };
        let mesh = Mesh::uv_sphere(params)?;
        compare_with_oracle(name, &mesh, case)?;
        checked += 1;
    }
    assert_eq!(checked, 2, "both UV sphere cases are required");
    Ok(())
}

fn compare_with_oracle(name: &str, mesh: &Mesh, case: &Value) -> TestResult {
    let mut expected_vertices = Vec::new();
    for point in case["vertices"].as_array().ok_or("vertices missing")? {
        let coords = point.as_array().ok_or("position must be an array")?;
        let mut position = [0.0_f64; 3];
        for (slot, value) in position.iter_mut().zip(coords) {
            *slot = value.as_f64().ok_or("coordinate must be a number")?;
        }
        expected_vertices.push(position);
    }
    let actual_vertices: Vec<[f64; 3]> = mesh
        .vertices
        .iter()
        .map(|vertex| vertex.co.to_array())
        .collect();
    assert_eq!(
        actual_vertices, expected_vertices,
        "{name} vertex order and positions"
    );

    let storage_index: HashMap<u32, usize> = mesh
        .vertices
        .iter()
        .enumerate()
        .map(|(index, vertex)| (vertex.id, index))
        .collect();
    let to_storage = |id: u32| {
        storage_index
            .get(&id)
            .copied()
            .ok_or("face or edge references an unknown vertex id")
    };

    let mut actual_edges = BTreeSet::new();
    for edge in &mesh.edges {
        let [a, b] = edge.vertices;
        actual_edges.insert(undirected(to_storage(a)?, to_storage(b)?));
    }
    let mut expected_edges = BTreeSet::new();
    for edge in case["edges"].as_array().ok_or("edges missing")? {
        let endpoints = oracle_indices(edge)?;
        if endpoints.len() != 2 {
            return Err("edge must have two endpoints".into());
        }
        expected_edges.insert(undirected(endpoints[0], endpoints[1]));
    }
    assert_eq!(mesh.edges.len(), expected_edges.len(), "{name} edge count");
    assert_eq!(actual_edges, expected_edges, "{name} undirected edge set");

    let mut actual_faces = Vec::with_capacity(mesh.faces.len());
    for face in &mesh.faces {
        let corners = face
            .vertices
            .iter()
            .map(|&id| to_storage(id))
            .collect::<Result<Vec<_>, _>>()?;
        actual_faces.push(corners);
    }
    actual_faces.sort();
    let mut expected_faces = Vec::new();
    for face in case["faces"].as_array().ok_or("faces missing")? {
        expected_faces.push(oracle_indices(face)?);
    }
    expected_faces.sort();
    assert_eq!(
        actual_faces, expected_faces,
        "{name} face set with loop starts"
    );
    Ok(())
}
