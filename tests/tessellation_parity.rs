// SPDX-License-Identifier: GPL-3.0-or-later
//
// Blender 5.2.2 `Mesh.loop_triangles` oracle for the shared `Mesh::tessellate` implementation.

use std::error::Error;

use glam::DVec3;
use potter_core::geom::Mesh;
use serde_json::Value;

type TestResult<T = ()> = Result<T, Box<dyn Error>>;

const ORACLE: &str = include_str!("fixtures/tessellation/blender_loop_triangles.json");

#[test]
fn tessellation_matches_blender_522_loop_triangles() -> TestResult {
    let oracle: Value = serde_json::from_str(ORACLE)?;
    assert_eq!(oracle["blender_version"], "5.2.2");
    let cases = oracle["cases"].as_array().ok_or("oracle cases missing")?;
    assert_eq!(
        cases.len(),
        140,
        "complete Blender tessellation oracle is required"
    );
    for case in cases {
        let name = case["kind"].as_str().ok_or("case kind missing")?;
        let positions = case["positions"]
            .as_array()
            .ok_or("case positions missing")?
            .iter()
            .map(|point| {
                let point = point.as_array().ok_or("position must be an array")?;
                Ok(DVec3::new(
                    point[0].as_f64().ok_or("x missing")?,
                    point[1].as_f64().ok_or("y missing")?,
                    point[2].as_f64().ok_or("z missing")?,
                ))
            })
            .collect::<TestResult<Vec<_>>>()?;
        let face = (0..positions.len()).collect::<Vec<_>>();
        let mesh = Mesh::from_positions_and_faces(positions, vec![face])?;
        let actual = mesh.tessellate()?;
        let expected = case["triangles"]
            .as_array()
            .ok_or("case triangles missing")?
            .iter()
            .map(|triangle| {
                let triangle = triangle.as_array().ok_or("triangle must be an array")?;
                Ok([
                    u32::try_from(triangle[0].as_u64().ok_or("triangle index missing")?)?,
                    u32::try_from(triangle[1].as_u64().ok_or("triangle index missing")?)?,
                    u32::try_from(triangle[2].as_u64().ok_or("triangle index missing")?)?,
                ])
            })
            .collect::<TestResult<Vec<_>>>()?;
        assert_eq!(actual, expected, "Blender tessellation case {name}");
    }
    Ok(())
}
