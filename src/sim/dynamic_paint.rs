//! Deterministic vertex colors from world-space dynamic-paint brush proximity.

use glam::{DMat4, DVec3};

use crate::{
    error::{ErrorCode, PotError, Result},
    geom::Mesh,
};

const MAX_DISTANCE_CHECKS: usize = 20_000_000;
const MAX_PAINT_VERTICES: usize = 1_000_000;
const MAX_BRUSH_POSITIONS: usize = 1_000_000;

/// Paint vertex colors using the strongest brush influence at each mesh vertex.
///
/// Results follow `canvas.vertices` storage order. Uncovered vertices are transparent
/// zero color; covered vertices retain `color` with alpha weighted by falloff and strength.
pub fn paint(
    canvas: &Mesh,
    canvas_world: DMat4,
    brushes: &[DVec3],
    radius: f64,
    color: [f64; 4],
    strength: f64,
) -> Result<Vec<[f64; 4]>> {
    if !canvas_world.is_finite() {
        return Err(invalid("dynamic paint canvas transform must be finite"));
    }
    if !radius.is_finite() || radius < 0.0 {
        return Err(invalid(
            "dynamic paint radius must be finite and non-negative",
        ));
    }
    if !strength.is_finite() || !(0.0..=1.0).contains(&strength) {
        return Err(invalid(
            "dynamic paint strength must be between zero and one",
        ));
    }
    if color.iter().any(|component| !component.is_finite()) {
        return Err(invalid("dynamic paint color must be finite"));
    }
    if canvas.vertices.len() > MAX_PAINT_VERTICES || brushes.len() > MAX_BRUSH_POSITIONS {
        return Err(limit(
            "dynamic paint is limited to one million vertices and brushes",
        ));
    }
    if brushes.iter().any(|brush| !brush.is_finite()) {
        return Err(invalid("dynamic paint brush positions must be finite"));
    }
    let checks = canvas
        .vertices
        .len()
        .checked_mul(brushes.len())
        .ok_or_else(|| limit("dynamic paint distance work exceeds limits"))?;
    if checks > MAX_DISTANCE_CHECKS {
        return Err(limit(
            "dynamic paint is limited to 20 million brush-vertex checks",
        ));
    }

    let mut output = Vec::with_capacity(canvas.vertices.len());
    for vertex in &canvas.vertices {
        if !vertex.co.is_finite() {
            return Err(invalid("dynamic paint canvas contains a non-finite vertex"));
        }
        let position = canvas_world.transform_point3(vertex.co);
        if !position.is_finite() {
            return Err(invalid(
                "dynamic paint transform produced a non-finite vertex",
            ));
        }
        let mut influence: f64 = 0.0;
        if strength > 0.0 {
            for brush in brushes {
                let offset = position - *brush;
                if !offset.is_finite() {
                    continue;
                }
                let scale = offset.abs().max_element();
                let falloff = if radius == 0.0 {
                    if scale == 0.0 { 1.0 } else { 0.0 }
                } else if scale > radius {
                    0.0
                } else {
                    let normalized_distance = (offset / radius).length();
                    if normalized_distance >= 1.0 {
                        0.0
                    } else {
                        let linear = 1.0 - normalized_distance;
                        linear * linear * (3.0 - 2.0 * linear)
                    }
                };
                influence = influence.max(falloff * strength);
                if influence >= strength {
                    break;
                }
            }
        }
        output.push(if influence > 0.0 {
            [color[0], color[1], color[2], color[3] * influence]
        } else {
            [0.0; 4]
        });
    }
    Ok(output)
}

fn invalid(message: &str) -> PotError {
    PotError::new(ErrorCode::InvalidArgument, message)
}

fn limit(message: &str) -> PotError {
    PotError::new(ErrorCode::LimitExceeded, message)
}

#[cfg(test)]
mod tests {
    #![expect(clippy::expect_used, reason = "solver tests use valid fixture inputs")]
    use glam::{DMat4, DVec3};

    use crate::geom::{Mesh, Vertex};

    use super::paint;

    #[test]
    fn world_space_brush_falloff_is_storage_ordered_and_bounded() {
        let canvas = Mesh {
            vertices: vec![
                Vertex {
                    id: 9,
                    co: DVec3::ZERO,
                },
                Vertex {
                    id: 2,
                    co: DVec3::X,
                },
                Vertex {
                    id: 7,
                    co: DVec3::new(2.0, 0.0, 0.0),
                },
            ],
            ..Mesh::default()
        };
        let colors = paint(
            &canvas,
            DMat4::from_translation(DVec3::new(10.0, 0.0, 0.0)),
            &[DVec3::new(10.0, 0.0, 0.0)],
            2.0,
            [0.2, 0.4, 0.6, 1.0],
            0.5,
        )
        .expect("painting succeeds");
        assert_eq!(colors.len(), 3);
        assert_eq!(colors[0], [0.2, 0.4, 0.6, 0.5]);
        assert!(colors[1][3] > 0.0 && colors[1][3] < colors[0][3]);
        assert_eq!(colors[2], [0.0; 4]);
        assert_eq!(
            paint(
                &canvas,
                DMat4::IDENTITY,
                &[],
                2.0,
                [0.2, 0.4, 0.6, 1.0],
                0.5
            )
            .expect("painting without brushes succeeds"),
            vec![[0.0; 4]; 3]
        );
    }
}
