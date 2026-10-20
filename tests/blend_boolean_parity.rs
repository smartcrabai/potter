//! Blender-gated parity for EXACT and FLOAT Booleans against the committed corpus in
//! `tests/fixtures/boolean_parity`.
//!
//! The corpus holds one Blender 5.2 expectation per case in `expected/<case>.json`, produced by
//! `generate_blender_expectations.py`. Regenerate it from the repository root (Blender must run with
//! `--factory-startup`):
//!
//! ```text
//! /Applications/Blender.app/Contents/MacOS/Blender -b --factory-startup \
//!     --python tests/fixtures/boolean_parity/generate_blender_expectations.py -- \
//!     tests/fixtures/boolean_parity
//! ```
//!
//! Commit the regenerated `expected/*.json`. `scene.blend` is written next to them and is git-ignored.
//!
//! Driver: the test regenerates `scene.blend` (every case in one scene) in a temporary directory and
//! imports it with `pot import`, which must report no losses. Each case evaluates its host with only
//! its own Boolean enabled. The committed expectations assert the recorded rings and payloads;
//! same-run Blender expectations provide complete edge endpoints and point/edge associations,
//! face-corner point values, all named UV layers and pins, attributes, colors, and derived normal vectors.
//!
//! All 53 EXACT and 11 FLOAT rows are asserted, including Blender FLOAT's defective outputs
//! (research §3.3, §3.4, §3.7). A missing Blender executable fails the test.

#[path = "common/blender_file.rs"]
mod blender_file;
use blender_file::blender_executable;

#[path = "fixtures/boolean_parity/comparator.rs"]
mod comparator;
use comparator::{
    TestResult, evaluate_boolean, expected_canonical, find_node, import_blend_scene, mismatches,
    run_generator,
};

use std::{fs, path::Path};

use serde_json::Value;
use tempfile::tempdir;

const BLENDER_REQUIRED: &str = "Blender is required for the Boolean parity corpus: install Blender 5.2, or set POTTER_BLENDER to its executable";

/// The six reviewer-verified EXACT cases (research §3.7). Each must be asserted.
const REVIEWED_CASES: [&str; 6] = [
    "attributes_difference_exact",
    "attributes_union_exact",
    "attributes_intersect_exact",
    "vertex_group_difference_exact",
    "vertex_group_union_exact",
    "material_transfer_union_exact",
];

/// Corpus rows with `solver == "EXACT"`.
const EXACT_ROWS: usize = 53;
/// Corpus rows with `solver == "FLOAT"`.
const FLOAT_ROWS: usize = 11;

#[test]
fn boolean_matches_blender_parity_corpus() -> TestResult {
    let blender = blender_executable().ok_or(BLENDER_REQUIRED)?;
    let corpus = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/boolean_parity");
    let directory = tempdir()?;
    let root = fs::canonicalize(directory.path())?;
    run_generator(
        &blender,
        &corpus.join("generate_blender_expectations.py"),
        &root,
    )?;
    let document = import_blend_scene(&blender, &root.join("scene.blend"), &root.join("project"))?;

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

    let mut asserted_exact = Vec::new();
    let mut asserted_float = Vec::new();
    let mut failures = Vec::new();
    for path in paths {
        let expected_json: Value = serde_json::from_slice(&fs::read(&path)?)?;
        let solver = expected_json["solver"]
            .as_str()
            .ok_or("Boolean case solver is missing")?;
        if solver != "EXACT" && solver != "FLOAT" {
            return Err(
                format!("unexpected Boolean solver `{solver}` in {}", path.display()).into(),
            );
        }
        let case = expected_json["case"]
            .as_str()
            .ok_or("Blender case name is missing")?
            .to_owned();
        let generated_path = root.join("expected").join(format!("{case}.json"));
        let generated_json: Value = serde_json::from_slice(&fs::read(&generated_path)?)?;
        if generated_json["case"].as_str() != Some(case.as_str()) {
            return Err(format!(
                "generated Blender expectation `{}` does not describe case `{case}`",
                generated_path.display()
            )
            .into());
        }
        if !expected_json["faces"].is_array() || !expected_json["vertex_groups"].is_object() {
            return Err(format!(
                "{solver} case `{case}` must record faces and vertex groups, including empty payloads"
            )
            .into());
        }
        let node_id = find_node(&document, &case)?;
        match evaluate_boolean(&document, &node_id) {
            Ok(actual) => {
                let expected = expected_canonical(&expected_json)?;
                failures.extend(
                    mismatches(&actual, &expected)
                        .into_iter()
                        .map(|difference| format!("{case}: {difference}")),
                );
                let runtime_expected = expected_canonical(&generated_json)?;
                failures.extend(
                    mismatches(&actual, &runtime_expected)
                        .into_iter()
                        .map(|difference| format!("{case}: runtime Blender record: {difference}")),
                );
            }
            Err(error) => failures.push(format!("{case}: {error}")),
        }
        match solver {
            "EXACT" => asserted_exact.push(case),
            "FLOAT" => asserted_float.push(case),
            _ => unreachable!("solver was checked above"),
        }
    }
    assert_eq!(
        asserted_exact.len(),
        EXACT_ROWS,
        "every EXACT corpus row must be asserted"
    );
    assert_eq!(
        asserted_float.len(),
        FLOAT_ROWS,
        "every FLOAT corpus row must be asserted"
    );
    for case in REVIEWED_CASES {
        assert!(
            asserted_exact.iter().any(|name| name.as_str() == case),
            "reviewed case `{case}` is not asserted"
        );
    }
    assert!(
        failures.is_empty(),
        "{} Boolean parity mismatches:\n{}",
        failures.len(),
        failures.join("\n")
    );
    Ok(())
}
