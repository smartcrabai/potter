use std::error::Error;

#[path = "common/blender_checked.rs"]
mod blender_checked;
#[path = "common/blender_script.rs"]
mod blender_script;
#[path = "common/process.rs"]
mod process;

use blender_checked::blender_executable;
use blender_script::run_blender_script;
use potter_core::{
    eval::EvaluationContext,
    exchange::{
        ExportOptions,
        blend::{export_blend, import_blend},
    },
    model::RenderSettings,
    ops,
    store::Project,
};
use serde_json::json;
use tempfile::tempdir;

const BUILD_FIXTURE: &str = r#"
import bpy
import sys
from pathlib import Path

if bpy.app.version != (5, 2, 2):
    raise RuntimeError(f"fixture requires Blender 5.2.2, found {bpy.app.version_string}")
root = Path(sys.argv[sys.argv.index("--") + 1])
bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
scene.render.engine = "BLENDER_EEVEE"
eevee = scene.eevee
eevee.volumetric_tile_size = "2"
eevee.volumetric_samples = 37
eevee.volumetric_sample_distribution = 0.35
eevee.use_volume_custom_range = False
eevee.volumetric_start = 0.25
eevee.volumetric_end = 50.0
bpy.ops.wm.save_as_mainfile(filepath=str(root / "source.blend"))
"#;

const READBACK: &str = r#"
import bpy
import sys
from pathlib import Path

if bpy.app.version != (5, 2, 2):
    raise RuntimeError(f"readback requires Blender 5.2.2, found {bpy.app.version_string}")
root = Path(sys.argv[sys.argv.index("--") + 1])
bpy.ops.wm.open_mainfile(filepath=str(root / "updated.blend"))
e = bpy.context.scene.eevee
assert e.volumetric_tile_size == "16", e.volumetric_tile_size
assert e.volumetric_samples == 201, e.volumetric_samples
assert abs(e.volumetric_sample_distribution - 0.65) < 1e-6, e.volumetric_sample_distribution
assert e.use_volume_custom_range is True, e.use_volume_custom_range
assert abs(e.volumetric_start - 1.25) < 1e-6, e.volumetric_start
assert abs(e.volumetric_end - 250.0) < 1e-6, e.volumetric_end
"#;

fn assert_volume_settings(render: &RenderSettings) {
    assert_eq!(render.eevee_volumetric_tile_size, 16);
    assert_eq!(render.eevee_volumetric_samples, 201);
    assert!((render.eevee_volumetric_sample_distribution - 0.65).abs() < 1e-6);
    assert!(render.eevee_use_volume_custom_range);
    assert!((render.eevee_volumetric_start - 1.25).abs() < 1e-6);
    assert!((render.eevee_volumetric_end - 250.0).abs() < 1e-6);
}

#[test]
fn eevee_volume_settings_import_update_export_and_reimport() -> Result<(), Box<dyn Error>> {
    let Some(blender) = blender_executable() else {
        eprintln!("skipping EEVEE volume settings roundtrip: Blender is unavailable");
        return Ok(());
    };
    let directory = tempdir()?;
    let root = directory.path();
    run_blender_script(
        &blender,
        "make_volume_settings.py",
        BUILD_FIXTURE,
        root,
        "Blender EEVEE volume settings fixture failed",
    )?;

    let imported = import_blend(&root.join("source.blend"), Some(blender.as_path()))?;
    let source_render = &imported.doc.scenes[&imported.doc.active_scene].render;
    assert_eq!(source_render.eevee_volumetric_tile_size, 2);
    assert_eq!(source_render.eevee_volumetric_samples, 37);
    assert!((source_render.eevee_volumetric_sample_distribution - 0.35).abs() < 1e-6);
    assert!(!source_render.eevee_use_volume_custom_range);
    assert!((source_render.eevee_volumetric_start - 0.25).abs() < 1e-6);
    assert!((source_render.eevee_volumetric_end - 50.0).abs() < 1e-6);

    let updated = ops::apply_batch(
        &imported.doc,
        &json!({
            "schema_version": 1,
            "base_revision": imported.doc.revision,
            "operations": [{
                "op": "render.update",
                "target": {"id": imported.doc.active_scene},
                "set": {
                    "eevee_volumetric_tile_size": 16,
                    "eevee_volumetric_samples": 201,
                    "eevee_volumetric_sample_distribution": 0.65,
                    "eevee_use_volume_custom_range": true,
                    "eevee_volumetric_start": 1.25,
                    "eevee_volumetric_end": 250.0
                }
            }]
        }),
    )?
    .doc;
    assert_volume_settings(&updated.scenes[&updated.active_scene].render);

    let project = Project::init(root.join("project"))?;
    let blend = root.join("updated.blend");
    let context = EvaluationContext::default();
    export_blend(
        &updated,
        &project,
        &blend,
        &ExportOptions {
            allow_lossy: true,
            pack: false,
            blender: Some(blender.as_path()),
            context: &context,
        },
    )?;
    run_blender_script(
        &blender,
        "read_volume_settings.py",
        READBACK,
        root,
        "Blender EEVEE volume settings readback failed",
    )?;

    let reimported = import_blend(&blend, Some(blender.as_path()))?;
    assert_volume_settings(&reimported.doc.scenes[&reimported.doc.active_scene].render);
    let serialized = serde_json::to_value(&updated.scenes[&updated.active_scene].render)?;
    for field in [
        "eevee_volumetric_tile_size",
        "eevee_volumetric_samples",
        "eevee_volumetric_sample_distribution",
        "eevee_use_volume_custom_range",
        "eevee_volumetric_start",
        "eevee_volumetric_end",
    ] {
        assert!(
            serialized.get(field).is_some(),
            "{field} should be serialized when non-default"
        );
    }
    Ok(())
}
