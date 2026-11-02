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
world = bpy.data.worlds.new("ProbeWorld")
world.sun_threshold = 3.125
world.sun_angle = 0.25
world.use_sun_shadow = False
world.sun_shadow_maximum_resolution = 0.004
world.sun_shadow_jitter_overblur = 37.5
world.sun_shadow_filter_radius = 2.5
world.use_sun_shadow_jitter = True
scene.world = world
local_grid_data = bpy.data.lightprobes.new("Local Grid", "VOLUME")
local_grid = bpy.data.objects.new("Local Grid", local_grid_data)
scene.collection.objects.link(local_grid)
eevee = scene.eevee
eevee.use_raytracing = True
scene.eevee.clamp_surface_indirect = 0.0
eevee.use_fast_gi = True
eevee.ray_tracing_method = "SCREEN"
raytracing = eevee.ray_tracing_options
raytracing.resolution_scale = "4"
raytracing.screen_trace_quality = 0.4
raytracing.screen_trace_thickness = 0.25
raytracing.trace_max_roughness = 0.75
raytracing.use_backface_hit = False
raytracing.backface_radiance_scale = 0.4
raytracing.use_denoise = True
raytracing.denoise_spatial = False
raytracing.denoise_temporal = True
raytracing.denoise_bilateral = False
eevee.fast_gi_bias = 0.2
eevee.fast_gi_resolution = "8"
eevee.fast_gi_step_count = 32
eevee.fast_gi_ray_count = 5
eevee.fast_gi_quality = 0.7
eevee.fast_gi_distance = 200.0
eevee.fast_gi_thickness_near = 0.3
eevee.fast_gi_method = "AMBIENT_OCCLUSION_ONLY"
raytracing_off = bpy.data.scenes.new("RaytracingOff")
raytracing_off.render.engine = "BLENDER_EEVEE"
raytracing_off.eevee.use_raytracing = False
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
assert bpy.context.scene.world is not None
assert abs(bpy.context.scene.world.sun_threshold - 3.125) < 1e-6, bpy.context.scene.world.sun_threshold
assert abs(bpy.context.scene.world.sun_angle - 0.25) < 1e-6
assert bpy.context.scene.world.use_sun_shadow is False
assert abs(bpy.context.scene.world.sun_shadow_maximum_resolution - 0.004) < 1e-6
assert abs(bpy.context.scene.world.sun_shadow_jitter_overblur - 37.5) < 1e-6
assert abs(bpy.context.scene.world.sun_shadow_filter_radius - 2.5) < 1e-6
assert bpy.context.scene.world.use_sun_shadow_jitter is True
assert abs(bpy.context.scene.eevee.clamp_surface_indirect - 12.5) < 1e-6
e = bpy.context.scene.eevee
assert e.volumetric_tile_size == "16", e.volumetric_tile_size
assert e.volumetric_samples == 201, e.volumetric_samples
assert abs(e.volumetric_sample_distribution - 0.65) < 1e-6, e.volumetric_sample_distribution
assert e.use_volume_custom_range is True, e.use_volume_custom_range
assert abs(e.volumetric_start - 1.25) < 1e-6, e.volumetric_start
assert abs(e.volumetric_end - 250.0) < 1e-6, e.volumetric_end
assert e.use_raytracing is False, e.use_raytracing
assert e.use_fast_gi is True, e.use_fast_gi
assert e.ray_tracing_method == "SCREEN", e.ray_tracing_method
r = e.ray_tracing_options
assert r.resolution_scale == "4", r.resolution_scale
assert abs(r.screen_trace_quality - 0.6) < 1e-6, r.screen_trace_quality
assert abs(r.screen_trace_thickness - 0.25) < 1e-6, r.screen_trace_thickness
assert abs(r.trace_max_roughness - 0.75) < 1e-6, r.trace_max_roughness
assert r.use_backface_hit is False, r.use_backface_hit
assert abs(r.backface_radiance_scale - 0.4) < 1e-6, r.backface_radiance_scale
assert r.use_denoise is True, r.use_denoise
assert r.denoise_spatial is False, r.denoise_spatial
assert r.denoise_temporal is True, r.denoise_temporal
assert r.denoise_bilateral is False, r.denoise_bilateral
assert abs(e.fast_gi_bias - 0.2) < 1e-6, e.fast_gi_bias
assert e.fast_gi_resolution == "8", e.fast_gi_resolution
assert e.fast_gi_step_count == 32, e.fast_gi_step_count
assert e.fast_gi_ray_count == 5, e.fast_gi_ray_count
assert abs(e.fast_gi_quality - 0.7) < 1e-6, e.fast_gi_quality
assert abs(e.fast_gi_distance - 200.0) < 1e-6, e.fast_gi_distance
assert abs(e.fast_gi_thickness_near - 0.3) < 1e-6, e.fast_gi_thickness_near
assert e.fast_gi_method == "AMBIENT_OCCLUSION_ONLY", e.fast_gi_method
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
    let source_scene = &imported.doc.scenes[&imported.doc.active_scene];
    assert_eq!(source_scene.eevee_has_local_volume_probe_grids, Some(true));
    assert!(imported.doc.scenes.values().any(|scene| {
        scene.name == "RaytracingOff" && scene.eevee_has_local_volume_probe_grids == Some(false)
    }));
    let source_render = &imported.doc.scenes[&imported.doc.active_scene].render;
    let source_world_id = imported.doc.scenes[&imported.doc.active_scene]
        .world
        .as_ref()
        .ok_or("imported scene world is missing")?;
    let source_world = &imported.doc.worlds[source_world_id];
    assert_eq!(source_world.eevee_sun_threshold, Some(f64::from(3.125_f32)));
    assert_eq!(source_world.eevee_sun_angle, Some(0.25));
    assert_eq!(source_world.eevee_use_sun_shadow, Some(false));
    assert_eq!(
        source_world.eevee_sun_shadow_maximum_resolution,
        Some(f64::from(0.004_f32))
    );
    assert_eq!(source_world.eevee_sun_shadow_jitter_overblur, Some(37.5));
    assert_eq!(source_world.eevee_sun_shadow_filter_radius, Some(2.5));
    assert_eq!(source_world.eevee_use_sun_shadow_jitter, Some(true));
    assert_eq!(source_render.eevee_clamp_surface_indirect, Some(0.0));
    assert_eq!(source_render.eevee_use_raytracing, Some(true));
    assert!(imported.doc.scenes.values().any(|scene| {
        scene.name == "RaytracingOff" && scene.render.eevee_use_raytracing == Some(false)
    }));
    assert_eq!(source_render.eevee_volumetric_tile_size, 2);
    assert_eq!(source_render.eevee_volumetric_samples, 37);
    assert_eq!(source_render.eevee_use_fast_gi, Some(true));
    assert_eq!(
        source_render.eevee_raytracing_method,
        Some(potter_core::model::EeveeRayTracingMethod::Screen)
    );
    assert_eq!(source_render.eevee_raytracing_resolution_scale, Some(4));
    assert_eq!(
        source_render.eevee_raytracing_screen_trace_quality,
        Some(f64::from(0.4_f32))
    );
    assert_eq!(
        source_render.eevee_raytracing_screen_trace_thickness,
        Some(0.25)
    );
    assert_eq!(
        source_render.eevee_raytracing_trace_max_roughness,
        Some(0.75)
    );
    assert_eq!(source_render.eevee_raytracing_use_backface_hit, Some(false));
    assert_eq!(
        source_render.eevee_raytracing_backface_radiance_scale,
        Some(f64::from(0.4_f32))
    );
    assert_eq!(source_render.eevee_raytracing_use_denoise, Some(true));
    assert_eq!(source_render.eevee_raytracing_denoise_spatial, Some(false));
    assert_eq!(source_render.eevee_raytracing_denoise_temporal, Some(true));
    assert_eq!(
        source_render.eevee_raytracing_denoise_bilateral,
        Some(false)
    );
    assert_eq!(source_render.eevee_fast_gi_bias, Some(f64::from(0.2_f32)));
    assert_eq!(source_render.eevee_fast_gi_resolution, Some(8));
    assert_eq!(source_render.eevee_fast_gi_step_count, Some(32));
    assert_eq!(source_render.eevee_fast_gi_ray_count, Some(5));
    assert_eq!(
        source_render.eevee_fast_gi_quality,
        Some(f64::from(0.7_f32))
    );
    assert_eq!(source_render.eevee_fast_gi_distance, Some(200.0));
    assert_eq!(
        source_render.eevee_fast_gi_thickness_near,
        Some(f64::from(0.3_f32))
    );
    assert_eq!(
        source_render.eevee_fast_gi_method,
        Some(potter_core::model::EeveeFastGiMethod::AmbientOcclusionOnly)
    );
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
                    "eevee_use_raytracing": false,
                    "eevee_volumetric_start": 1.25,
                    "eevee_raytracing_screen_trace_quality": 0.6,
                    "eevee_volumetric_end": 250.0,
                    "eevee_clamp_surface_indirect": 12.5
                }
            }]
        }),
    )?
    .doc;
    assert_volume_settings(&updated.scenes[&updated.active_scene].render);
    assert_eq!(
        updated.scenes[&updated.active_scene]
            .render
            .eevee_use_raytracing,
        Some(false)
    );

    assert_eq!(
        updated.scenes[&updated.active_scene]
            .render
            .eevee_raytracing_screen_trace_quality,
        Some(0.6)
    );
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
    let reimported_world_id = reimported.doc.scenes[&reimported.doc.active_scene]
        .world
        .as_ref()
        .ok_or("round-tripped world is missing")?;
    let reimported_world = &reimported.doc.worlds[reimported_world_id];
    assert_eq!(
        reimported_world.eevee_sun_threshold,
        Some(f64::from(3.125_f32))
    );
    assert_eq!(reimported_world.eevee_sun_angle, Some(0.25));
    assert_eq!(reimported_world.eevee_use_sun_shadow, Some(false));
    assert_eq!(
        reimported_world.eevee_sun_shadow_maximum_resolution,
        Some(f64::from(0.004_f32))
    );
    assert_eq!(
        reimported_world.eevee_sun_shadow_jitter_overblur,
        Some(37.5)
    );
    assert_eq!(reimported_world.eevee_sun_shadow_filter_radius, Some(2.5));
    assert_eq!(reimported_world.eevee_use_sun_shadow_jitter, Some(true));
    assert_eq!(
        reimported.doc.scenes[&reimported.doc.active_scene]
            .render
            .eevee_clamp_surface_indirect,
        Some(f64::from(12.5_f32))
    );
    assert_volume_settings(&reimported.doc.scenes[&reimported.doc.active_scene].render);
    assert_eq!(
        reimported.doc.scenes[&reimported.doc.active_scene]
            .render
            .eevee_use_raytracing,
        Some(false)
    );
    assert_eq!(
        reimported.doc.scenes[&reimported.doc.active_scene]
            .render
            .eevee_raytracing_screen_trace_quality,
        Some(f64::from(0.6_f32))
    );
    let serialized = serde_json::to_value(&updated.scenes[&updated.active_scene].render)?;
    for field in [
        "eevee_use_raytracing",
        "eevee_use_fast_gi",
        "eevee_raytracing_method",
        "eevee_raytracing_resolution_scale",
        "eevee_raytracing_screen_trace_quality",
        "eevee_raytracing_screen_trace_thickness",
        "eevee_raytracing_trace_max_roughness",
        "eevee_raytracing_use_backface_hit",
        "eevee_raytracing_backface_radiance_scale",
        "eevee_raytracing_use_denoise",
        "eevee_raytracing_denoise_spatial",
        "eevee_raytracing_denoise_temporal",
        "eevee_raytracing_denoise_bilateral",
        "eevee_fast_gi_bias",
        "eevee_fast_gi_resolution",
        "eevee_fast_gi_step_count",
        "eevee_fast_gi_ray_count",
        "eevee_fast_gi_quality",
        "eevee_fast_gi_distance",
        "eevee_fast_gi_thickness_near",
        "eevee_fast_gi_method",
    ] {
        assert!(
            serialized.get(field).is_some(),
            "{field} should be serialized when non-default"
        );
    }
    Ok(())
}
