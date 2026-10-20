//! Regression: a self-crossing quad (bowtie) previews. The quad is a plane whose `v2` and `v3`
//! swap places, so its boundary crosses itself and its Newell area is zero. Before preview read
//! `Mesh::tessellate`, the ear clipper rejected it with `could not triangulate face 0`, while
//! Blender renders the same face from its looptris. The triangle-level oracle is
//! `tests/tessellation_parity.rs`, which includes a bowtie quad and self-intersecting n-gons.

use std::{error::Error, fs, process::Output};

use serde_json::{Value, json};
use tempfile::tempdir;

type TestResult<T = ()> = Result<T, Box<dyn Error>>;

fn pot() -> std::process::Command {
    std::process::Command::new(env!("CARGO_BIN_EXE_pot"))
}

#[test]
fn self_crossing_quad_previews_in_every_view() -> TestResult {
    let directory = tempdir()?;
    let scene = directory.path().join("scene");
    let init: Output = pot().arg("init").arg(&scene).arg("--json").output()?;
    assert!(init.status.success(), "pot init failed");

    // Corners of `plane`: v0 (-1,-1), v1 (1,-1), v2 (1,1), v3 (-1,1). Swapping v2 and v3 makes the
    // quad 0-1-2-3 cross itself along its two diagonals.
    let batch = directory.path().join("batch.json");
    fs::write(
        &batch,
        json!({
            "schema_version": 1,
            "base_revision": 0,
            "operations": [
                {"op": "node.create", "id": "bow", "kind": "plane", "params": {"size": 2.0}},
                {
                    "op": "mesh.transform_elements",
                    "target": {"id": "bow", "elements": {"domain": "vertex", "ids": ["v2"]}},
                    "translation": [-2.0, 0.0, 0.0]
                },
                {
                    "op": "mesh.transform_elements",
                    "target": {"id": "bow", "elements": {"domain": "vertex", "ids": ["v3"]}},
                    "translation": [2.0, 0.0, 0.0]
                }
            ]
        })
        .to_string(),
    )?;
    let apply: Output = pot()
        .arg("apply")
        .arg(&scene)
        .arg("--file")
        .arg(&batch)
        .arg("--json")
        .output()?;
    assert!(
        apply.status.success(),
        "pot apply failed: {}",
        String::from_utf8_lossy(&apply.stdout)
    );

    let out = directory.path().join("preview");
    let preview: Output = pot()
        .arg("preview")
        .arg(&scene)
        .args(["--views", "front,top", "--mode", "solid", "--size", "64"])
        .arg("--out")
        .arg(&out)
        .arg("--json")
        .output()?;
    let stdout = String::from_utf8(preview.stdout)?;
    assert!(
        preview.status.success(),
        "pot preview failed: {stdout}\nstderr: {}",
        String::from_utf8_lossy(&preview.stderr)
    );
    let envelope: Value = serde_json::from_str(&stdout)?;
    let previews = envelope["result"]["previews"]
        .as_array()
        .ok_or("preview result has no previews")?;
    assert_eq!(previews.len(), 2, "one preview per requested view");
    Ok(())
}
