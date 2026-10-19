// SPDX-License-Identifier: GPL-3.0-or-later
//
// Blender 5.2.2 `primitive_cube_add` storage-order oracle for Potter's `box` primitive.

use std::{collections::HashMap, error::Error};

use potter_core::geom::{self, Mesh};
use serde_json::{Value, json};

type BoxError = Box<dyn Error>;
type TestResult = Result<(), BoxError>;

const ORACLE: &str = include_str!("fixtures/primitive_box/box_order.json");

/// Storage-order topology: positions, and edge, face and loop references as indices into the
/// vertex list. Nothing is sorted or canonicalised, so the order itself is what is compared.
#[derive(Debug, PartialEq)]
struct Topology {
    vertices: Vec<[f64; 3]>,
    edges: Vec<[usize; 2]>,
    faces: Vec<Vec<usize>>,
    loops: Vec<usize>,
}

fn potter_topology(mesh: &Mesh) -> Result<Topology, BoxError> {
    let index_of: HashMap<u32, usize> = mesh
        .vertices
        .iter()
        .enumerate()
        .map(|(index, vertex)| (vertex.id, index))
        .collect();
    let index = |id: u32| -> Result<usize, BoxError> {
        index_of
            .get(&id)
            .copied()
            .ok_or_else(|| "vertex ID has no storage index".into())
    };
    let mut faces = Vec::with_capacity(mesh.faces.len());
    for face in &mesh.faces {
        let mut corners = Vec::with_capacity(face.vertices.len());
        for &id in &face.vertices {
            corners.push(index(id)?);
        }
        faces.push(corners);
    }
    let mut edges = Vec::with_capacity(mesh.edges.len());
    for edge in &mesh.edges {
        edges.push([index(edge.vertices[0])?, index(edge.vertices[1])?]);
    }
    Ok(Topology {
        vertices: mesh
            .vertices
            .iter()
            .map(|vertex| vertex.co.to_array())
            .collect(),
        edges,
        loops: faces.iter().flatten().copied().collect(),
        faces,
    })
}

fn indices(value: &Value) -> Result<Vec<usize>, BoxError> {
    let mut result = Vec::new();
    for item in value.as_array().ok_or("indices must be an array")? {
        let raw = item.as_u64().ok_or("index must be an integer")?;
        result.push(usize::try_from(raw)?);
    }
    Ok(result)
}

fn position(value: &Value) -> Result<[f64; 3], BoxError> {
    let point = value
        .as_array()
        .filter(|point| point.len() == 3)
        .ok_or("position must have three coordinates")?;
    Ok([
        point[0].as_f64().ok_or("x missing")?,
        point[1].as_f64().ok_or("y missing")?,
        point[2].as_f64().ok_or("z missing")?,
    ])
}

fn oracle_topology(value: &Value) -> Result<Topology, BoxError> {
    let mut vertices = Vec::new();
    for point in value["vertices"].as_array().ok_or("vertices missing")? {
        vertices.push(position(point)?);
    }
    let mut edges = Vec::new();
    for edge in value["edges"].as_array().ok_or("edges missing")? {
        let endpoints = indices(edge)?;
        if endpoints.len() != 2 {
            return Err("edge must have two endpoints".into());
        }
        edges.push([endpoints[0], endpoints[1]]);
    }
    let mut faces = Vec::new();
    for face in value["faces"].as_array().ok_or("faces missing")? {
        faces.push(indices(face)?);
    }
    Ok(Topology {
        vertices,
        edges,
        faces,
        loops: indices(&value["loops"])?,
    })
}

#[test]
fn box_storage_order_matches_blender_primitive_cube_add() -> TestResult {
    let oracle: Value = serde_json::from_str(ORACLE)?;
    assert_eq!(oracle["blender_version"], "5.2.2");
    let cases = oracle["cases"].as_object().ok_or("oracle cases missing")?;
    assert_eq!(cases.len(), 2, "both box sizes are required");
    for (name, case) in cases {
        let size = case["size"].as_f64().ok_or("case size missing")?;
        let mesh = geom::primitive("box", &json!({ "size": size }))?;
        let expected = oracle_topology(&case["plain"])?;
        assert_eq!(potter_topology(&mesh)?, expected, "{name} storage order");
        assert_eq!(
            oracle_topology(&case["bmesh_round_trip"])?,
            expected,
            "{name} BMesh round trip"
        );
    }
    Ok(())
}
