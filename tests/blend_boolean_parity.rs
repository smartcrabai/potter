//! Blender-gated parity for the EXACT Boolean against the committed corpus in
//! `tests/fixtures/boolean_parity`.
//!
//! The corpus holds one Blender 5.2 expectation per case in `expected/<case>.json`, produced
//! by `generate_blender_expectations.py`. Regenerate it from the repository root (Blender must
//! run with `--factory-startup`):
//!
//! ```text
//! /Applications/Blender.app/Contents/MacOS/Blender -b --factory-startup \
//!     --python tests/fixtures/boolean_parity/generate_blender_expectations.py -- \
//!     tests/fixtures/boolean_parity
//! ```
//!
//! Commit the regenerated `expected/*.json`. `scene.blend` is written next to them and is
//! git-ignored.
//!
//! Driver: the generator runs into a temporary directory to produce `scene.blend` (every case
//! in one scene). `pot import` brings that scene in, and each EXACT case evaluates its host with
//! only its own Boolean enabled. The result is compared with the committed expectation.
//! Blender gating follows the other `blend_*` tests: skipped without Blender, and a panic when
//! `POTTER_REQUIRE_BLENDER` is set.
//!
//! FLOAT cases are in the corpus as Blender's defective outputs (research §3.3, §3.4, §3.7
//! INDEX row). They are `PR17b` targets for the `FLOAT` port, so this driver does not assert them.
//! The open-operand EXACT cases from §3.5 assert both `hole_tolerant` settings and all operations.
//! Comparator, applied to both Potter and Blender results:
//! - vertex and edge counts must match exactly;
//! - each face is `(ring, material_index)`. The ring is quantized to 1e-6, rotated to start
//!   at its minimum vertex, and the face list is sorted, so vertex order, ring start, and face
//!   order are ignored while per-face materials are checked;
//! - the signed volume (fan from each ring's first vertex, divided by six) must match with
//!   relative tolerance 1e-6, or absolute tolerance 1e-9 when the expected volume is zero.
//!
//! Custom face attributes (`fid`) and UV layers are recorded in the corpus for the attribute
//! transfer slice. This driver does not compare them. The import reports the custom mesh
//! attribute as a loss; any other import loss fails the test.

use std::{collections::HashMap, error::Error, fs, path::Path, process::Command};

#[path = "common/blender_file.rs"]
mod blender_file;
use blender_file::blender_executable;

use glam::DVec3;
use potter_core::{
    eval::{EvaluationContext, Snapshot},
    geom::Mesh,
    model::SceneDoc,
};
use serde_json::Value;
use tempfile::tempdir;

type TestResult<T> = Result<T, Box<dyn Error>>;

/// Coordinates are quantized to this many steps per unit (1e-6 resolution).
const COORDINATE_SCALE: f64 = 1.0e6;
const VOLUME_RELATIVE_TOLERANCE: f64 = 1.0e-6;
const VOLUME_ABSOLUTE_TOLERANCE: f64 = 1.0e-9;

/// Quantized ring (rotated to its minimum vertex) and material slot.
type CanonicalFace = (Vec<[i64; 3]>, u64);

/// Comparator view of one evaluated Boolean result.
#[derive(Debug)]
struct Canonical {
    vertices: usize,
    edges: usize,
    faces: Vec<CanonicalFace>,
    volume: f64,
}

fn run_pot(args: &[&str]) -> TestResult<Value> {
    let output = Command::new(env!("CARGO_BIN_EXE_pot"))
        .args(args)
        .output()?;
    assert!(
        output.status.success(),
        "pot {} failed: stdout={} stderr={}",
        args.first().copied().unwrap_or("command"),
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    Ok(serde_json::from_slice(&output.stdout)?)
}

fn quantize(point: DVec3) -> [i64; 3] {
    let [x, y, z] = (point * COORDINATE_SCALE).round().to_array();
    [x as i64, y as i64, z as i64]
}

fn canonical_face(ring: &[DVec3], material: u64) -> CanonicalFace {
    let mut quantized: Vec<[i64; 3]> = ring.iter().map(|point| quantize(*point)).collect();
    let start = quantized
        .iter()
        .enumerate()
        .min_by_key(|&(_, point)| *point)
        .map_or(0, |(index, _)| index);
    quantized.rotate_left(start);
    (quantized, material)
}

fn signed_volume(rings: &[Vec<DVec3>]) -> f64 {
    rings
        .iter()
        .map(|ring| {
            ring.split_first().map_or(0.0, |(first, rest)| {
                rest.windows(2)
                    .map(|pair| first.dot(pair[0].cross(pair[1])) / 6.0)
                    .sum::<f64>()
            })
        })
        .sum()
}

fn potter_canonical(mesh: &Mesh) -> TestResult<Canonical> {
    let positions: HashMap<u32, DVec3> = mesh
        .vertices
        .iter()
        .map(|vertex| (vertex.id, vertex.co))
        .collect();
    let mut faces = Vec::with_capacity(mesh.faces.len());
    let mut rings = Vec::with_capacity(mesh.faces.len());
    for face in &mesh.faces {
        let ring = face
            .vertices
            .iter()
            .map(|id| {
                positions
                    .get(id)
                    .copied()
                    .ok_or("Potter face references a missing vertex")
            })
            .collect::<Result<Vec<_>, _>>()?;
        faces.push(canonical_face(&ring, u64::from(face.material_index)));
        rings.push(ring);
    }
    faces.sort();
    Ok(Canonical {
        vertices: mesh.vertices.len(),
        edges: mesh.edges.len(),
        faces,
        volume: signed_volume(&rings),
    })
}

fn point_from_json(value: &Value) -> TestResult<DVec3> {
    let coordinates = value
        .as_array()
        .ok_or("Blender vertex must be an array")?
        .iter()
        .map(|coordinate| {
            coordinate
                .as_f64()
                .ok_or("Blender vertex coordinate is invalid")
        })
        .collect::<Result<Vec<f64>, _>>()?;
    match coordinates.as_slice() {
        [x, y, z] => Ok(DVec3::new(*x, *y, *z)),
        _ => Err("Blender vertex must have three coordinates".into()),
    }
}

fn count_from_json(value: &Value, name: &str) -> TestResult<usize> {
    let count = value
        .as_u64()
        .ok_or_else(|| format!("Blender {name} count is invalid"))?;
    Ok(usize::try_from(count)?)
}

fn expected_canonical(expected: &Value) -> TestResult<Canonical> {
    let mut faces = Vec::new();
    for face in expected["faces"]
        .as_array()
        .ok_or("Blender faces are missing")?
    {
        let ring = face["ring"]
            .as_array()
            .ok_or("Blender face ring is missing")?
            .iter()
            .map(point_from_json)
            .collect::<TestResult<Vec<_>>>()?;
        let material = face["material_index"]
            .as_u64()
            .ok_or("Blender material index is invalid")?;
        faces.push(canonical_face(&ring, material));
    }
    faces.sort();
    Ok(Canonical {
        vertices: count_from_json(&expected["counts"]["vertices"], "vertex")?,
        edges: count_from_json(&expected["counts"]["edges"], "edge")?,
        faces,
        volume: expected["volume"]
            .as_f64()
            .ok_or("Blender volume is missing")?,
    })
}

fn assert_volume(actual: f64, expected: f64, case: &str) {
    let matches = if expected.abs() <= VOLUME_ABSOLUTE_TOLERANCE {
        actual.abs() <= VOLUME_ABSOLUTE_TOLERANCE
    } else {
        ((actual - expected) / expected).abs() <= VOLUME_RELATIVE_TOLERANCE
    };
    assert!(
        matches,
        "{case}: volume {actual:.9} differs from Blender {expected:.9}"
    );
}

fn assert_case(document: &SceneDoc, expected_json: &Value) -> TestResult<()> {
    let case = expected_json["case"]
        .as_str()
        .ok_or("Blender case name is missing")?;
    let node_id = document
        .nodes
        .iter()
        .find(|(_, node)| node.name == case)
        .map(|(id, _)| id.clone())
        .ok_or_else(|| format!("Blender host `{case}` is missing from the imported scene"))?;
    let mut boolean_only = document.clone();
    for (id, node) in &mut boolean_only.nodes {
        for modifier in &mut node.modifiers {
            modifier.enabled = *id == node_id && modifier.modifier_type == "boolean";
        }
    }
    let snapshot = Snapshot::evaluate(&boolean_only, &EvaluationContext::default())
        .map_err(|error| format!("{case}: {error}"))?;
    let mesh = snapshot
        .meshes
        .get(&node_id)
        .ok_or_else(|| format!("Potter did not evaluate `{case}`"))?;
    let actual = potter_canonical(mesh)?;
    let expected = expected_canonical(expected_json)?;

    assert_eq!(actual.vertices, expected.vertices, "{case}: vertex count");
    assert_eq!(actual.edges, expected.edges, "{case}: edge count");
    assert_eq!(
        actual.faces.len(),
        expected.faces.len(),
        "{case}: face count"
    );
    assert_eq!(
        actual.faces, expected.faces,
        "{case}: canonical rings or per-face materials differ"
    );
    assert_volume(actual.volume, expected.volume, case);
    Ok(())
}

#[test]
fn exact_boolean_matches_blender_parity_corpus() -> TestResult<()> {
    let Some(blender) = blender_executable() else {
        eprintln!("skipping Boolean parity test: Blender is unavailable");
        return Ok(());
    };
    let corpus = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/boolean_parity");
    let directory = tempdir()?;
    let root = fs::canonicalize(directory.path())?;
    let blender_output = Command::new(&blender)
        .args(["--background", "--factory-startup", "--python"])
        .arg(corpus.join("generate_blender_expectations.py"))
        .arg("--")
        .arg(&root)
        .output()?;
    assert!(
        blender_output.status.success(),
        "Blender Boolean parity generator failed: stdout={} stderr={}",
        String::from_utf8_lossy(&blender_output.stdout),
        String::from_utf8_lossy(&blender_output.stderr)
    );

    let source_path = root
        .join("scene.blend")
        .to_str()
        .ok_or("Blend path is not UTF-8")?
        .to_owned();
    let project = root.join("project");
    let project_path = project.to_str().ok_or("project path is not UTF-8")?;
    let blender_path = blender
        .to_str()
        .ok_or("Blender path is not UTF-8")?
        .to_owned();
    run_pot(&["init", project_path])?;
    let imported = run_pot(&[
        "import",
        project_path,
        "--file",
        &source_path,
        "--format",
        "blend",
        "--mode",
        "replace",
        "--base-revision",
        "0",
        "--blender",
        &blender_path,
    ])?;
    let losses = imported["result"]["losses"]
        .as_array()
        .ok_or("import result has no losses array")?;
    assert!(
        losses
            .iter()
            .all(|loss| loss["feature_id"] == "blender.mesh.attribute"),
        "unexpected import losses: {losses:?}"
    );
    let document: SceneDoc = serde_json::from_slice(&fs::read(project.join("scene.json"))?)?;

    let mut paths = Vec::new();
    for entry in fs::read_dir(corpus.join("expected"))? {
        let path = entry?.path();
        if path
            .extension()
            .is_some_and(|extension| extension == "json")
        {
            paths.push(path);
        }
    }
    paths.sort();
    assert!(!paths.is_empty(), "boolean parity corpus has no cases");
    let mut open_asserted = 0;
    let mut asserted = 0;
    for path in paths {
        let expected_json: Value = serde_json::from_slice(&fs::read(&path)?)?;
        // FLOAT cases are PR17b targets (FAST port). They stay in the corpus for that work and
        // are not asserted here.
        if expected_json["solver"] != "EXACT" {
            continue;
        }
        assert_case(&document, &expected_json)?;
        asserted += 1;
        if expected_json["research_section"] == "3.5" {
            open_asserted += 1;
        }
    }
    assert!(asserted > 0, "no EXACT cases were asserted");
    assert_eq!(
        open_asserted, 18,
        "expected all 18 open-operand EXACT cases to be asserted"
    );
    Ok(())
}
