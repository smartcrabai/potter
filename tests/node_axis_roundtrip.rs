//! Object `track_axis` and `up_axis` set through node operations survive `.blend` export and import.
//!
//! The axes are created and then updated through Potter's own operations. The exported `.blend`
//! is read back in Blender 5.2.x, which must report the same `Object.track_axis` and `up_axis`,
//! and re-importing it must restore the typed node fields. Requires Blender 5.2.x; skipped when it
//! is missing, and failing instead when `POTTER_REQUIRE_BLENDER` is set.
use std::{env, error::Error, fs, path::PathBuf, process::Command};

#[path = "common/blender_checked.rs"]
mod blender_checked;
#[path = "common/process.rs"]
mod process;

use blender_checked::blender_executable;
use potter_core::{
    eval::EvaluationContext,
    exchange::{
        ExportOptions,
        blend::{export_blend, import_blend},
    },
    model::{Id, SceneDoc, TrackAxis, UpAxis},
    ops,
    store::Project,
};
use serde_json::{Value, json};
use tempfile::tempdir;

type TestResult<T> = Result<T, Box<dyn Error>>;

const SCENE_ID: &str = "00000000-0000-4000-8000-000000000000";
const OBJECT: &str = "axis";

/// Reads each object's `track_axis` and `up_axis` from a saved `.blend`.
const READBACK: &str = r#"
import bpy
import json
import sys

argv = sys.argv[sys.argv.index("--") + 1:]
bpy.ops.wm.open_mainfile(filepath=argv[0])
obj = bpy.data.objects[argv[1]]
with open(argv[2], "w", encoding="utf-8") as output:
    json.dump({"track_axis": obj.track_axis, "up_axis": obj.up_axis}, output)
"#;

fn require_blender() -> bool {
    env::var_os("POTTER_REQUIRE_BLENDER").is_some_and(|value| !value.is_empty())
}

/// Returns Blender 5.2.x, or `None` (skip) when it is missing.
fn blender_5_2() -> TestResult<Option<PathBuf>> {
    let Some(blender) = blender_executable() else {
        assert!(
            !require_blender(),
            "POTTER_REQUIRE_BLENDER is set but Blender is not available"
        );
        eprintln!("skipping node axis round trip: Blender is unavailable");
        return Ok(None);
    };
    let mut command = Command::new(&blender);
    command.arg("--version");
    let output = process::run_guarded(command)?;
    let version = String::from_utf8_lossy(&output.stdout)
        .lines()
        .next()
        .unwrap_or_default()
        .to_owned();
    if version.starts_with("Blender 5.2") {
        return Ok(Some(blender));
    }
    assert!(
        !require_blender(),
        "node axis round trip needs Blender 5.2.x, found `{version}`"
    );
    eprintln!("skipping node axis round trip: found `{version}`, need Blender 5.2.x");
    Ok(None)
}

fn apply(doc: &SceneDoc, operation: &Value) -> TestResult<SceneDoc> {
    let batch = json!({
        "schema_version": 1,
        "base_revision": doc.revision,
        "operations": [operation],
    });
    Ok(ops::apply_batch(doc, &batch)?.doc)
}

fn node_axes(doc: &SceneDoc, id: &str) -> TestResult<(TrackAxis, UpAxis)> {
    let node = doc
        .nodes
        .get(&Id::new(id)?)
        .ok_or("axis test node is missing")?;
    Ok((node.track_axis, node.up_axis))
}

#[test]
fn node_axes_are_set_validated_and_round_trip_through_blend() -> TestResult<()> {
    let Some(blender) = blender_5_2()? else {
        return Ok(());
    };
    let directory = tempdir()?;
    let root = directory.path();

    // Created with non-default axes, then updated through `set`.
    let doc = apply(
        &SceneDoc::new(SCENE_ID.to_owned()),
        &json!({
            "op": "node.create",
            "id": OBJECT,
            "kind": "box",
            "params": {"size": 2.0},
            "track_axis": "NEG_X",
            "up_axis": "Y",
        }),
    )?;
    assert_eq!(node_axes(&doc, OBJECT)?, (TrackAxis::NegX, UpAxis::Y));

    // Unknown identifiers are refused rather than defaulted.
    assert!(
        apply(
            &doc,
            &json!({"op": "node.update", "target": {"id": OBJECT}, "set": {"track_axis": "POS_W"}}),
        )
        .is_err(),
        "an unknown track_axis must fail validation"
    );
    assert!(
        apply(
            &doc,
            &json!({"op": "node.update", "target": {"id": OBJECT}, "set": {"up_axis": "NEG_Z"}}),
        )
        .is_err(),
        "Blender's up_axis has no negative identifiers"
    );

    let doc = apply(
        &doc,
        &json!({
            "op": "node.update",
            "target": {"id": OBJECT},
            "set": {"track_axis": "POS_Z"},
        }),
    )?;
    assert_eq!(node_axes(&doc, OBJECT)?, (TrackAxis::PosZ, UpAxis::Y));

    let project = Project::init(root.join("project"))?;
    let context = EvaluationContext::default();
    let blend = root.join("axes.blend");
    export_blend(
        &doc,
        &project,
        &blend,
        &ExportOptions {
            allow_lossy: false,
            pack: false,
            blender: Some(blender.as_path()),
            context: &context,
        },
    )?;

    let script = root.join("readback_axes.py");
    fs::write(&script, READBACK)?;
    let readback_path = root.join("readback_axes.json");
    let output = Command::new(&blender)
        .args(["--background", "--factory-startup", "--python"])
        .arg(&script)
        .arg("--")
        .arg(&blend)
        .arg(OBJECT)
        .arg(&readback_path)
        .output()?;
    assert!(
        output.status.success(),
        "Blender axis readback failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let readback: Value = serde_json::from_slice(&fs::read(&readback_path)?)?;
    assert_eq!(readback["track_axis"], "POS_Z", "Blender track_axis");
    assert_eq!(readback["up_axis"], "Y", "Blender up_axis");

    let imported = import_blend(&blend, Some(blender.as_path()))?;
    let imported_id = imported
        .id_mappings
        .get(format!("Object:{OBJECT}"))
        .and_then(Value::as_str)
        .ok_or("Blender axis object ID mapping is missing")?;
    assert_eq!(
        node_axes(&imported.doc, imported_id)?,
        (TrackAxis::PosZ, UpAxis::Y),
        "re-imported axes must match the exported ones"
    );
    Ok(())
}
