// SPDX-License-Identifier: GPL-3.0-or-later
//
// Blender 5.2.2 `calc_uvs` oracle for the primitives' `UVMap` layer. The oracle stores each loop's
// `[u, v]` in Blender's convention (v up) in face order; Potter stores `v` flipped (v down).

use std::error::Error;

use glam::DVec3;
use potter_core::geom::{
    BoxParams, CircleFillType, CircleParams, ConeParams, CylinderParams, EndFillType, GridParams,
    IcosphereParams, Mesh, PlaneParams, TorusParams,
};
use serde_json::Value;

type TestResult = Result<(), Box<dyn Error>>;

const ORACLE: &str = include_str!("fixtures/primitive_uvs/primitive_uv_oracle.json");
const TOLERANCE: f64 = 1.0e-5;

fn oracle_uv(cases: &Value, name: &str) -> Result<Vec<[f64; 2]>, Box<dyn Error>> {
    let mut result = Vec::new();
    for pair in cases[name]["uv"].as_array().ok_or("uv missing")? {
        let values = pair.as_array().ok_or("uv pair must be an array")?;
        result.push([
            values[0].as_f64().ok_or("u missing")?,
            values[1].as_f64().ok_or("v missing")?,
        ]);
    }
    Ok(result)
}

/// The `UVMap` layer's loop values in the mesh's storage face order, as Potter stores them.
fn stored_uv(mesh: &Mesh) -> Result<Vec<[f64; 2]>, Box<dyn Error>> {
    let store = mesh
        .attributes
        .get("uv_layers")
        .ok_or("primitive has no uv_layers store")?;
    let layer = &store["layers"][0];
    assert_eq!(layer["name"], "UVMap");
    let entries = layer["faces"].as_array().ok_or("layer faces missing")?;
    let mut result = Vec::new();
    for face in &mesh.faces {
        let entry = entries
            .iter()
            .find(|entry| entry["face_id"] == face.id)
            .ok_or("face has no UV entry")?;
        for pair in entry["uv"].as_array().ok_or("uv missing")? {
            result.push([
                pair[0].as_f64().ok_or("u missing")?,
                pair[1].as_f64().ok_or("v missing")?,
            ]);
        }
    }
    Ok(result)
}

fn assert_uv(name: &str, mesh: &Mesh, cases: &Value) -> TestResult {
    let expected = oracle_uv(cases, name)?;
    let actual = stored_uv(mesh)?;
    assert_eq!(actual.len(), expected.len(), "{name} loop count");
    for (index, (stored, blender)) in actual.iter().zip(&expected).enumerate() {
        // Potter stores v down: stored v = 1 - Blender v.
        let error = (stored[0] - blender[0])
            .abs()
            .max((stored[1] - (1.0 - blender[1])).abs());
        assert!(
            error <= TOLERANCE,
            "{name} loop {index}: stored {stored:?}, Blender {blender:?}"
        );
    }
    Ok(())
}

fn oracle() -> Result<Value, Box<dyn Error>> {
    let root: Value = serde_json::from_str(ORACLE)?;
    assert_eq!(root["blender_version"], "5.2.2");
    Ok(root["cases"].clone())
}

#[test]
fn box_plane_and_grid_uvs_match_blender() -> TestResult {
    let cases = oracle()?;
    let cube = Mesh::box_mesh(BoxParams {
        size: DVec3::splat(2.0),
    })?;
    assert_uv("box_size_2", &cube, &cases)?;
    assert_uv(
        "plane_size_2",
        &Mesh::plane(PlaneParams { size: 2.0 })?,
        &cases,
    )?;
    let grid = Mesh::grid(GridParams {
        size_x: 2.0,
        size_y: 2.0,
        x_subdivisions: 3,
        y_subdivisions: 3,
    })?;
    assert_uv("grid_2x2", &grid, &cases)
}

#[test]
fn circle_uvs_match_blender() -> TestResult {
    let cases = oracle()?;
    let open = Mesh::circle(CircleParams {
        vertices: 8,
        radius: 1.0,
        fill_type: CircleFillType::Nothing,
    })?;
    assert_uv("circle_8", &open, &cases)?;
    let ngon = Mesh::circle(CircleParams {
        vertices: 5,
        radius: 0.75,
        fill_type: CircleFillType::Ngon,
    })?;
    assert_uv("circle_ngon_5", &ngon, &cases)?;
    let fan = Mesh::circle(CircleParams {
        vertices: 5,
        radius: 0.75,
        fill_type: CircleFillType::TriFan,
    })?;
    assert_uv("circle_trifan_5", &fan, &cases)
}

#[test]
fn cylinder_and_cone_uvs_match_blender() -> TestResult {
    let cases = oracle()?;
    let cylinder = |vertices, end_fill_type| {
        Mesh::cylinder(CylinderParams {
            vertices,
            radius: 1.0,
            depth: 2.0,
            end_fill_type,
        })
    };
    assert_uv("cylinder_8", &cylinder(8, EndFillType::Ngon)?, &cases)?;
    assert_uv(
        "cylinder_default",
        &cylinder(32, EndFillType::Ngon)?,
        &cases,
    )?;
    assert_uv(
        "cylinder_8_trifan",
        &cylinder(8, EndFillType::TriFan)?,
        &cases,
    )?;
    assert_uv(
        "cylinder_8_nothing",
        &cylinder(8, EndFillType::Nothing)?,
        &cases,
    )?;
    let cone = |vertices, radius1, radius2, end_fill_type| {
        Mesh::cone(ConeParams {
            vertices,
            radius1,
            radius2,
            depth: 2.0,
            end_fill_type,
        })
    };
    assert_uv("cone_8", &cone(8, 1.0, 0.0, EndFillType::Ngon)?, &cases)?;
    assert_uv(
        "cone_default",
        &cone(32, 1.0, 0.0, EndFillType::Ngon)?,
        &cases,
    )?;
    assert_uv(
        "cone_8_trifan",
        &cone(8, 1.0, 0.0, EndFillType::TriFan)?,
        &cases,
    )?;
    assert_uv(
        "cone_8_nothing",
        &cone(8, 1.0, 0.0, EndFillType::Nothing)?,
        &cases,
    )?;
    assert_uv(
        "cone_trunc_8",
        &cone(8, 1.0, 0.5, EndFillType::Ngon)?,
        &cases,
    )?;
    assert_uv(
        "cone_inverted_8",
        &cone(8, 0.0, 1.0, EndFillType::Ngon)?,
        &cases,
    )?;
    assert_uv(
        "cone_inverted_32",
        &cone(32, 0.0, 1.0, EndFillType::Ngon)?,
        &cases,
    )
}

#[test]
fn torus_and_icosphere_uvs_match_blender() -> TestResult {
    let cases = oracle()?;
    let torus = |major_segments, minor_segments| {
        Mesh::torus(TorusParams {
            major_segments,
            minor_segments,
            major_radius: 1.0,
            minor_radius: 0.25,
            ..TorusParams::default()
        })
    };
    assert_uv("torus_8x4", &torus(8, 4)?, &cases)?;
    assert_uv("torus_default", &torus(48, 12)?, &cases)?;
    for (name, subdivisions) in [
        ("ico_sphere_1", 1),
        ("ico_sphere_default", 2),
        ("ico_sphere_3", 3),
        ("ico_sphere_4", 4),
    ] {
        let icosphere = Mesh::icosphere(IcosphereParams {
            subdivisions,
            radius: 1.0,
        })?;
        assert_uv(name, &icosphere, &cases)?;
    }
    Ok(())
}

/// Per-face UVs keyed by each face's loop list (vertex indices in loop order). Blender's UV sphere
/// order varies between processes, but a face's loop list and its loop values do not.
fn keyed_faces(
    faces: &[Vec<usize>],
    uv: &[[f64; 2]],
) -> Result<std::collections::BTreeMap<Vec<usize>, Vec<[f64; 2]>>, Box<dyn Error>> {
    let mut result = std::collections::BTreeMap::new();
    let mut offset = 0;
    for face in faces {
        let values = uv
            .get(offset..offset + face.len())
            .ok_or("uv has fewer loops than faces")?
            .to_vec();
        offset += face.len();
        result.insert(face.clone(), values);
    }
    Ok(result)
}

/// Blender's UV sphere face list, as vertex indices in loop order.
fn oracle_faces(case: &Value) -> Result<Vec<Vec<usize>>, Box<dyn Error>> {
    let mut faces = Vec::new();
    for face in case["faces"].as_array().ok_or("faces missing")? {
        let mut corners = Vec::new();
        for item in face.as_array().ok_or("face must be an array")? {
            corners.push(usize::try_from(
                item.as_u64().ok_or("index must be an integer")?,
            )?);
        }
        faces.push(corners);
    }
    Ok(faces)
}

/// Potter's face list as vertex indices in storage order and loop order.
fn stored_faces(mesh: &Mesh) -> Result<Vec<Vec<usize>>, Box<dyn Error>> {
    let index = |id: u32| -> Result<usize, Box<dyn Error>> {
        mesh.vertices
            .iter()
            .position(|vertex| vertex.id == id)
            .ok_or_else(|| "vertex ID has no storage index".into())
    };
    let mut faces = Vec::new();
    for face in &mesh.faces {
        let mut corners = Vec::new();
        for &id in &face.vertices {
            corners.push(index(id)?);
        }
        faces.push(corners);
    }
    Ok(faces)
}

#[test]
fn uv_sphere_uvs_match_blender() -> TestResult {
    let cases = oracle()?;
    for (name, segments, ring_count) in [("uv_sphere_8x4", 8, 4), ("uv_sphere_default", 32, 16)] {
        let mesh = Mesh::uv_sphere(potter_core::geom::UvSphereParams {
            segments,
            ring_count,
            radius: 1.0,
        })?;
        let expected = keyed_faces(&oracle_faces(&cases[name])?, &oracle_uv(&cases, name)?)?;
        let actual_uv = stored_uv(&mesh)?;
        let actual = keyed_faces(&stored_faces(&mesh)?, &actual_uv)?;
        assert_eq!(actual.len(), expected.len(), "{name} face count");
        for (faces, blender) in &expected {
            let stored = actual
                .get(faces)
                .ok_or_else(|| format!("{name}: Blender face {faces:?} missing from Potter"))?;
            for (index, (stored, blender)) in stored.iter().zip(blender).enumerate() {
                let error = (stored[0] - blender[0])
                    .abs()
                    .max((stored[1] - (1.0 - blender[1])).abs());
                assert!(
                    error <= TOLERANCE,
                    "{name} face {faces:?} loop {index}: stored {stored:?}, Blender {blender:?}"
                );
            }
        }
    }
    Ok(())
}
