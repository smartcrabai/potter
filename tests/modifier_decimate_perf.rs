//! Timing for Decimate COLLAPSE (ratio 0.1) on a 100k-triangle wavy grid, the same mesh as
//! `tests/fixtures/pr15_decimate/perf.py` times in Blender. Ignored by default; run with
//! `cargo test --release --test modifier_decimate_perf -- --ignored --nocapture`.

use std::{error::Error, time::Instant};

use glam::DVec3;
use potter_core::{
    geom::{Mesh, modifiers::evaluate_modifiers},
    model::{Id, Modifier, ModifierRuntime},
};
use serde_json::json;

type TestResult<T = ()> = Result<T, Box<dyn Error>>;

/// `N * N` vertices and `2 * (N - 1)^2` triangles, matching `perf.py`.
const N: usize = 224;

fn wavy_grid() -> TestResult<Mesh> {
    let mut positions = Vec::with_capacity(N * N);
    for j in 0..N {
        for i in 0..N {
            let x = -1.0 + 2.0 * i as f64 / (N - 1) as f64;
            let y = -1.0 + 2.0 * j as f64 / (N - 1) as f64;
            positions.push(DVec3::new(x, y, 0.12 * (x * 2.0).cos() * (y * 2.0).sin()));
        }
    }
    let mut faces = Vec::with_capacity(2 * (N - 1) * (N - 1));
    for j in 0..N - 1 {
        for i in 0..N - 1 {
            let a = j * N + i;
            faces.push(vec![a, a + 1, a + N + 1]);
            faces.push(vec![a, a + N + 1, a + N]);
        }
    }
    Ok(Mesh::from_positions_and_faces(positions, faces)?)
}

#[test]
#[ignore = "timing harness; run in release with --ignored --nocapture"]
fn decimate_collapse_100k_triangles_timing() -> TestResult {
    let mesh = wavy_grid()?;
    let modifier = Modifier {
        id: Id::new("decimate_perf")?,
        modifier_type: "decimate".to_owned(),
        name: "decimate".to_owned(),
        enabled: true,
        params: json!({"decimate_type": "COLLAPSE", "ratio": 0.1})
            .as_object()
            .ok_or("params")?
            .clone(),
        binding_data: None,
        runtime: ModifierRuntime::default(),
    };
    let mut best = f64::INFINITY;
    let mut faces_out = 0;
    for _ in 0..3 {
        let start = Instant::now();
        let output = evaluate_modifiers(&mesh, std::slice::from_ref(&modifier))?;
        best = best.min(start.elapsed().as_secs_f64());
        faces_out = output.faces.len();
    }
    eprintln!(
        "PERF potter triangles_in={} faces_out={faces_out} best_seconds={best:.4}",
        mesh.faces.len()
    );
    Ok(())
}
