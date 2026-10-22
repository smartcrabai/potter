#![expect(
    clippy::unwrap_used,
    reason = "Blender integration fixtures use fixed valid data"
)]

//! Blender 5.2.2 oracle for the particle emission schedule and texture influences.
//!
//! `tests/fixtures/particle_texture/generate.py` writes the scene and Blender's per-particle values.
//! Potter imports the scene and bakes the same frames. Each case's particles must match Blender's
//! birth frame, size, and lifetime to `VALUE_TOLERANCE`: Blender stores float32 values below 10, so
//! the rounding is below `1e-6`, and the bound leaves room for one float32 step. A density case must
//! keep the same number of particles that Blender keeps. Each particle's birth position must match
//! Blender's `location`, which is its birth point because physics is off; face and volume emitters
//! take their birth points from Blender's distribution (`particle_distribute.cc`).

use std::{cmp::Ordering, error::Error, fs, path::Path, process::Command};

#[path = "common/blender_file.rs"]
mod blender_file;
#[path = "common/process.rs"]
mod process;

use blender_file::blender_executable;
use serde_json::Value;
use tempfile::tempdir;

const VALUE_TOLERANCE: f64 = 1.0e-5;

/// One particle's birth frame, size, and lifetime.
#[derive(Clone, Copy, Debug)]
struct Sample {
    birth_frame: f64,
    size: f64,
    lifetime: f64,
}

fn run_ok(command: Command, operation: &str) -> Result<(), Box<dyn Error>> {
    let output = process::run_guarded(command)?;
    assert!(
        output.status.success(),
        "{operation} failed:\n{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    Ok(())
}

fn sorted_samples(samples: &mut [Sample]) {
    samples.sort_by(|left, right| left.birth_frame.total_cmp(&right.birth_frame));
}

fn assert_samples_match(case: &str, potter: &mut [Sample], blender: &mut [Sample]) {
    sorted_samples(potter);
    sorted_samples(blender);
    assert_eq!(
        potter.len(),
        blender.len(),
        "{case}: particle count differs"
    );
    for (index, (potter, blender)) in potter.iter().zip(blender.iter()).enumerate() {
        for (name, potter_value, blender_value) in [
            ("birth frame", potter.birth_frame, blender.birth_frame),
            ("size", potter.size, blender.size),
            ("lifetime", potter.lifetime, blender.lifetime),
        ] {
            assert!(
                (potter_value - blender_value).abs() <= VALUE_TOLERANCE,
                "{case} particle {index} {name}: potter={potter_value}, blender={blender_value}"
            );
        }
    }
}

fn blender_samples(case: &Value) -> Vec<Sample> {
    case["particles"]
        .as_array()
        .unwrap()
        .iter()
        .map(|particle| Sample {
            birth_frame: particle["birth_frame"].as_f64().unwrap(),
            size: particle["size"].as_f64().unwrap(),
            lifetime: particle["lifetime"].as_f64().unwrap(),
        })
        .collect()
}

/// The three numbers of a JSON array.
fn vector3(value: &Value) -> [f64; 3] {
    let values = value.as_array().unwrap();
    [
        values[0].as_f64().unwrap(),
        values[1].as_f64().unwrap(),
        values[2].as_f64().unwrap(),
    ]
}

/// Lexicographic order of two points.
fn compare_points(left: &[f64; 3], right: &[f64; 3]) -> Ordering {
    left.iter()
        .zip(right)
        .map(|(left, right)| left.total_cmp(right))
        .find(|order| *order != Ordering::Equal)
        .unwrap_or(Ordering::Equal)
}

/// Birth positions as sorted sets. Both sides list particles in index order; sorting keeps the
/// check independent of how either side orders particles that share a birth.
fn assert_positions_match(case: &str, potter: &[Value], blender: &Value) {
    let mut potter_points: Vec<[f64; 3]> = potter
        .iter()
        .map(|state| vector3(&state["birth_position"]))
        .collect();
    let mut blender_points: Vec<[f64; 3]> = blender
        .as_array()
        .unwrap()
        .iter()
        .map(|particle| vector3(&particle["location"]))
        .collect();
    potter_points.sort_by(compare_points);
    blender_points.sort_by(compare_points);
    assert_eq!(
        potter_points.len(),
        blender_points.len(),
        "{case}: particle count differs"
    );
    for (index, (potter_point, blender_point)) in
        potter_points.iter().zip(&blender_points).enumerate()
    {
        for (axis, (potter_value, blender_value)) in
            potter_point.iter().zip(blender_point).enumerate()
        {
            assert!(
                (potter_value - blender_value).abs() <= VALUE_TOLERANCE,
                "{case} particle {index} birth position axis {axis}: potter={potter_value}, blender={blender_value}"
            );
        }
    }
}

#[test]
fn particle_texture_influences_and_count_schedule_match_blender() -> Result<(), Box<dyn Error>> {
    let Some(blender) = blender_executable() else {
        eprintln!("skipping particle texture oracle; Blender is unavailable");
        return Ok(());
    };
    let directory = tempdir()?;
    let root = directory.path();
    let generator =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/particle_texture/generate.py");

    let mut generate = Command::new(&blender);
    generate
        .args([
            "--background",
            "--factory-startup",
            "--disable-autoexec",
            "--python-exit-code",
            "3",
            "--python",
        ])
        .arg(&generator)
        .arg("--")
        .arg(root);
    run_ok(generate, "Blender particle texture oracle")?;
    let expected: Value = serde_json::from_slice(&fs::read(root.join("blender_expected.json"))?)?;

    let project = root.join("project");
    let mut init = Command::new(env!("CARGO_BIN_EXE_pot"));
    init.args(["init"]).arg(&project).arg("--json");
    run_ok(init, "project initialization")?;
    let mut import = Command::new(env!("CARGO_BIN_EXE_pot"));
    import
        .args(["import"])
        .arg(&project)
        .args(["--file"])
        .arg(root.join("particle_texture.blend"))
        .args([
            "--format",
            "blend",
            "--mode",
            "replace",
            "--base-revision",
            "0",
            "--blender",
        ])
        .arg(&blender)
        .arg("--json");
    run_ok(import, "Blender import")?;

    let out = root.join("bake");
    let mut bake = Command::new(env!("CARGO_BIN_EXE_pot"));
    bake.arg("bake")
        .arg(&project)
        .args(["--kind", "simulation", "--frames", "1:21", "--out"])
        .arg(&out)
        .arg("--json");
    run_ok(bake, "simulation bake")?;

    let scene: Value = serde_json::from_slice(&fs::read(project.join("scene.json"))?)?;
    let nodes = scene["nodes"].as_object().unwrap();
    for (name, case) in expected.as_object().unwrap() {
        let node_id = nodes
            .iter()
            .find(|(_, node)| node["name"] == name.as_str())
            .map_or_else(
                || panic!("{name}: node is missing from the imported scene"),
                |(id, _)| id.clone(),
            );
        if let Some(explode) = case.get("explode") {
            // Explode cases compare the evaluated mesh counts of the imported node at the scene frame.
            let mut inspect = Command::new(env!("CARGO_BIN_EXE_pot"));
            inspect
                .arg("inspect")
                .arg(&project)
                .arg("--id")
                .arg(&node_id)
                .arg("--json");
            let output = process::run_guarded(inspect)?;
            assert!(
                output.status.success(),
                "{name}: inspect failed:\n{}",
                String::from_utf8_lossy(&output.stdout)
            );
            let envelope: Value = serde_json::from_slice(&output.stdout)?;
            let geometry = &envelope["result"]["items"][0]["evaluated_geometry"];
            assert_eq!(
                geometry["vertex_count"], explode["vertex_count"],
                "{name}: Explode vertex count"
            );
            assert_eq!(
                geometry["face_count"], explode["face_count"],
                "{name}: Explode face count"
            );
            continue;
        }
        let frame_index = case["frame"].as_u64().unwrap() - 1;
        let frame: Value =
            serde_json::from_slice(&fs::read(out.join(format!("frame_{frame_index:08}.json")))?)?;
        let states = frame["particles"][&node_id].as_array().unwrap();
        let mut potter: Vec<Sample> = states
            .iter()
            .map(|state| {
                let birth_frame = state["birth_frame"].as_f64().unwrap();
                Sample {
                    birth_frame,
                    size: state["size"].as_f64().unwrap(),
                    lifetime: state["death_frame"].as_f64().unwrap() - birth_frame,
                }
            })
            .collect();
        let mut blender = blender_samples(case);
        assert_samples_match(name, &mut potter, &mut blender);
        assert_positions_match(name, states, &case["particles"]);
    }
    Ok(())
}
