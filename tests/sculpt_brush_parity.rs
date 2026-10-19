//! Blender 5.2.2 parity for the dab-sequential sculpt brushes: Clay Strips, Inflate, and Layer.
//!
//! The fixtures come from `tests/fixtures/sculpt_brushes/generate_oracle.py`, which drives Blender
//! in a session with a display (the script's header explains why). Each fixture holds the base mesh,
//! the stroke's dabs, and Blender's resulting vertex positions. Blender computes in single precision;
//! the largest difference measured against these fixtures is 7.1e-7 (Clay Strips), so the tolerance
//! of 5e-6 leaves about a sevenfold margin.

use std::error::Error;

use glam::DVec3;
use serde_json::Value;

use potter_core::geom::{
    Mesh,
    sculpt::{self, Brush, Falloff, Stroke, StrokeSample},
};

const TOLERANCE: f64 = 5e-6;

const FIXTURES: &[(&str, &str)] = &[
    (
        "inflate_sphere",
        include_str!("fixtures/sculpt_brushes/inflate_sphere.json"),
    ),
    (
        "inflate_sphere_strength",
        include_str!("fixtures/sculpt_brushes/inflate_sphere_strength.json"),
    ),
    (
        "inflate_sphere_mirror",
        include_str!("fixtures/sculpt_brushes/inflate_sphere_mirror.json"),
    ),
    (
        "layer_sphere",
        include_str!("fixtures/sculpt_brushes/layer_sphere.json"),
    ),
    (
        "layer_sphere_clamp",
        include_str!("fixtures/sculpt_brushes/layer_sphere_clamp.json"),
    ),
    (
        "layer_sphere_mirror",
        include_str!("fixtures/sculpt_brushes/layer_sphere_mirror.json"),
    ),
    (
        "clay_strips_grid",
        include_str!("fixtures/sculpt_brushes/clay_strips_grid.json"),
    ),
    (
        "clay_strips_grid_mirror",
        include_str!("fixtures/sculpt_brushes/clay_strips_grid_mirror.json"),
    ),
    (
        "clay_strips_grid_pressure",
        include_str!("fixtures/sculpt_brushes/clay_strips_grid_pressure.json"),
    ),
];

fn number(value: &Value) -> Result<f64, Box<dyn Error>> {
    value
        .as_f64()
        .ok_or_else(|| format!("expected a number, found {value}").into())
}

fn vec3(value: &Value) -> Result<DVec3, Box<dyn Error>> {
    match value.as_array().map(Vec::as_slice) {
        Some([x, y, z]) => Ok(DVec3::new(number(x)?, number(y)?, number(z)?)),
        _ => Err(format!("expected a 3-vector, found {value}").into()),
    }
}

fn array<T>(
    value: &Value,
    mut parse: impl FnMut(&Value) -> Result<T, Box<dyn Error>>,
) -> Result<Vec<T>, Box<dyn Error>> {
    value
        .as_array()
        .ok_or_else(|| format!("expected an array, found {value}"))?
        .iter()
        .map(&mut parse)
        .collect()
}

/// Replays one fixture's stroke on its base mesh and returns the largest per-component difference
/// from Blender's result, after checking that the stroke moved something.
fn max_error_against_blender(name: &str, text: &str) -> Result<f64, Box<dyn Error>> {
    let document: Value = serde_json::from_str(text)?;
    let brush = match document["brush"].as_str() {
        Some("clay_strips") => Brush::ClayStrips,
        Some("inflate") => Brush::Inflate,
        Some("layer") => Brush::Layer,
        other => return Err(format!("{name}: unknown brush {other:?}").into()),
    };
    // Blender's Clay Strips brush uses the smoother falloff preset; the others use smooth.
    let falloff = if matches!(brush, Brush::ClayStrips) {
        Falloff::Smoother
    } else {
        Falloff::Smooth
    };
    let symmetry = if document["mirror"].as_bool() == Some(true) {
        vec!["x".to_owned()]
    } else {
        Vec::new()
    };
    let base = array(&document["base"], vec3)?;
    let faces = array(&document["faces"], |face| {
        array(face, |index| {
            let index = number(index)?;
            if index < 0.0 || index.fract() != 0.0 {
                return Err(format!("{name}: invalid vertex index {index}").into());
            }
            Ok(index as usize)
        })
    })?;
    let result = array(&document["result"], vec3)?;
    let dabs = array(&document["dabs"], |dab| {
        Ok((
            vec3(&dab["position"])?,
            number(&dab["pressure"])?,
            number(&dab["radius"])?,
            number(&dab["strength"])?,
        ))
    })?;
    let samples = dabs
        .into_iter()
        .enumerate()
        .map(
            |(index, (position, pressure, radius, strength))| StrokeSample {
                position,
                pressure,
                radius,
                strength,
                time: index as f64,
            },
        )
        .collect();
    let stroke = Stroke {
        brush,
        samples,
        falloff,
        symmetry,
        delta: None,
    };
    let mut mesh = Mesh::from_positions_and_faces(base.clone(), faces)?;
    sculpt::apply_stroke(&mut mesh, &stroke)?;
    if mesh.vertices.len() != result.len() {
        return Err(format!("{name}: vertex count changed").into());
    }
    let moved = mesh
        .vertices
        .iter()
        .zip(&base)
        .any(|(vertex, before)| vertex.co != *before);
    if !moved {
        return Err(format!("{name}: the stroke moved no vertex").into());
    }
    Ok(mesh
        .vertices
        .iter()
        .zip(&result)
        .map(|(vertex, expected)| (vertex.co - *expected).abs().max_element())
        .fold(0.0, f64::max))
}

#[test]
fn dab_brushes_match_blender_5_2_2() -> Result<(), Box<dyn Error>> {
    for (name, text) in FIXTURES {
        let error = max_error_against_blender(name, text)?;
        assert!(
            error <= TOLERANCE,
            "{name}: max vertex error {error:e} exceeds {TOLERANCE:e}"
        );
    }
    Ok(())
}
