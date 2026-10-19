//! Object ray visibility through `.blend`: Blender writes the flags, Potter imports them into each
//! node's `ray_visibility`, exports them back, and Blender reads the exported file.
//!
//! `ray_visibility_build.py` writes the source scene. A flag set to its Blender default is not
//! stored, so the default lamp and the default mesh carry no `ray_visibility` key in `scene.json`.
//! Blender reads every file with `ray_visibility_dump.py`, the same reader for the source and for
//! the Potter export.

#[path = "common/blender_checked.rs"]
mod blender_checked;
#[path = "common/native_oracle.rs"]
#[expect(
    dead_code,
    reason = "each test binary uses a subset of the shared oracle helpers"
)]
mod native_oracle;
#[path = "common/process.rs"]
mod process;

use std::{fs, path::Path};

use native_oracle::{
    TestResult, export, import_blend, loss_features, node_id, read_scene, run_blender_arguments,
    write_script,
};
use serde_json::{Value, json};
use tempfile::tempdir;

const BUILD_SCRIPT: &str = include_str!("fixtures/native_oracle/ray_visibility_build.py");
const DUMP_SCRIPT: &str = include_str!("fixtures/native_oracle/ray_visibility_dump.py");

/// Blender's `visible_*` property names, paired with the keys of `Node.ray_visibility`.
const FLAGS: [(&str, &str); 6] = [
    ("visible_camera", "camera"),
    ("visible_diffuse", "diffuse"),
    ("visible_glossy", "glossy"),
    ("visible_transmission", "transmission"),
    ("visible_volume_scatter", "volume_scatter"),
    ("visible_shadow", "shadow"),
];

const OBJECTS: [&str; 4] = ["HiddenFromCamera", "GlossyTransmissionOff", "Lamp", "Plain"];

/// The six flags Blender must hold for each object of `ray_visibility_build.py`.
fn expected_flags() -> Value {
    json!({
        "HiddenFromCamera": {
            "visible_camera": false,
            "visible_diffuse": true,
            "visible_glossy": true,
            "visible_transmission": true,
            "visible_volume_scatter": true,
            "visible_shadow": false,
        },
        "GlossyTransmissionOff": {
            "visible_camera": true,
            "visible_diffuse": true,
            "visible_glossy": false,
            "visible_transmission": false,
            "visible_volume_scatter": true,
            "visible_shadow": true,
        },
        "Lamp": {
            "visible_camera": false,
            "visible_diffuse": true,
            "visible_glossy": true,
            "visible_transmission": true,
            "visible_volume_scatter": true,
            "visible_shadow": true,
        },
        "Plain": {
            "visible_camera": true,
            "visible_diffuse": true,
            "visible_glossy": true,
            "visible_transmission": true,
            "visible_volume_scatter": true,
            "visible_shadow": true,
        },
    })
}

/// The `ray_visibility` object Potter stores for the Blender flags in `flags`.
fn stored_visibility(flags: &Value) -> Value {
    Value::Object(
        FLAGS
            .iter()
            .map(|(blender, potter)| ((*potter).to_owned(), flags[*blender].clone()))
            .collect(),
    )
}

/// Blender's six `visible_*` flags of every object in `blend`, as `ray_visibility_dump.py` reports.
fn dump_flags(blender: &Path, root: &Path, blend: &Path, name: &str) -> TestResult<Value> {
    let script = write_script(root, "ray_visibility_dump.py", DUMP_SCRIPT)?;
    let out = root.join(format!("{name}.json"));
    run_blender_arguments(blender, &script, Some(blend), &[out.as_path()])?;
    Ok(serde_json::from_slice(&fs::read(out)?)?)
}

#[test]
fn object_ray_visibility_survives_import_export_and_reads_back_in_blender() -> TestResult {
    let Some(blender) = blender_checked::blender_executable() else {
        return Ok(());
    };
    let root = tempdir()?;
    let build = write_script(root.path(), "ray_visibility_build.py", BUILD_SCRIPT)?;
    let source = root.path().join("source.blend");
    run_blender_arguments(&blender, &build, None, &[source.as_path()])?;
    let expected = expected_flags();
    assert_eq!(
        dump_flags(&blender, root.path(), &source, "source")?,
        expected,
        "the build script must write the flags it names"
    );

    let (project, report) = import_blend(&blender, root.path(), "project", &source)?;
    assert!(
        loss_features(&report)
            .iter()
            .all(|feature| !feature.contains("visib")),
        "ray visibility is representable, so it must not be reported as a loss: {report}"
    );
    let scene = read_scene(&project)?;
    for name in ["HiddenFromCamera", "GlossyTransmissionOff"] {
        let id = node_id(&scene, name)?;
        assert_eq!(
            scene["nodes"][&id]["ray_visibility"],
            stored_visibility(&expected[name]),
            "{name} must store the flags Blender holds, as non-default values"
        );
    }
    for name in ["Lamp", "Plain"] {
        let id = node_id(&scene, name)?;
        assert!(
            scene["nodes"][&id].get("ray_visibility").is_none(),
            "{name} has only default flags, so scene.json must not store ray_visibility: {}",
            scene["nodes"][&id]
        );
    }

    let exported = root.path().join("roundtrip.blend");
    export(&blender, &project, "blend", &exported)?;
    let reread = dump_flags(&blender, root.path(), &exported, "exported")?;
    for name in OBJECTS {
        assert_eq!(
            reread[name], expected[name],
            "{name}: Blender must read back the flags after Potter's export"
        );
    }
    Ok(())
}
