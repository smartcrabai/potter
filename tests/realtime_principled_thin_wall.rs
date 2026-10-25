//! EEVEE thin-wall glass and backlit subsurface on Blender 5.2.2 macOS arm64.
//!
//! `fixtures/native_oracle/eevee_thin_wall.py` renders a glass pane and a cutout leaf, each with no object thickness or
//! a shallow closed object. The two pane captures are compared to `--engine realtime` over every pixel. The leaf captures
//! remain committed EEVEE references, but realtime emits `UNSUPPORTED_FEATURE` for thin-wall subsurface lighting rather
//! than claiming pixel parity; Cycles' native thin-wall subsurface path has its own oracle.
//!
//! The per-channel pane bound, 1.6e-2, reuses the existing `realtime_glass_direct` EEVEE model bound (twice its measured
//! maximum 8.0e-3 closure-model deviation). Blender's captures use 64 EEVEE TAA samples; Potter's render is deterministic.
//! Blender-gated: with `POTTER_REQUIRE_BLENDER` set, a missing Blender fails the test.

#[path = "common/blender_checked.rs"]
mod blender_checked;
#[path = "common/native_oracle.rs"]
#[expect(
    dead_code,
    reason = "the shared oracle helpers include support for unrelated tests"
)]
mod native_oracle;
#[path = "common/process.rs"]
mod process;

use std::path::Path;

use native_oracle::{
    TestResult, apply_operations, checked, import_blend, pot, read_scene, run_blender_arguments,
    write_script,
};
use potter_core::image::{ImageData, ImageInterpolation, load_image_data, sample};
use potter_core::model::Image;
use serde_json::{Value, json};
use tempfile::tempdir;

const EEVEE_SCRIPT: &str = include_str!("fixtures/native_oracle/eevee_thin_wall.py");
const EEVEE_CAPTURES: [(&str, &[u8], bool); 4] = [
    (
        "pane_none",
        include_bytes!("fixtures/native_oracle/eevee_macos/thin_wall_pane_none.f32le"),
        false,
    ),
    (
        "pane_sphere",
        include_bytes!("fixtures/native_oracle/eevee_macos/thin_wall_pane_sphere.f32le"),
        false,
    ),
    (
        "leaf_none",
        include_bytes!("fixtures/native_oracle/eevee_macos/thin_wall_leaf_none.f32le"),
        true,
    ),
    (
        "leaf_sphere",
        include_bytes!("fixtures/native_oracle/eevee_macos/thin_wall_leaf_sphere.f32le"),
        true,
    ),
];

const SIZE: u32 = 96;
const ORTHO_SCALE: f64 = 2.4;
const PIXEL_TOLERANCE: f64 = 1.6e-2;

#[test]
fn realtime_principled_thin_wall_matches_eevee() -> TestResult {
    let Some(blender) = blender_checked::blender_executable() else {
        return Ok(());
    };
    let root = tempdir()?;
    let script = write_script(root.path(), "eevee_thin_wall.py", EEVEE_SCRIPT)?;

    for (case, capture, leaf) in EEVEE_CAPTURES {
        let blend = root.path().join(format!("{case}.blend"));
        run_blender_arguments(
            &blender,
            &script,
            None,
            &[Path::new(case), &blend, Path::new("-")],
        )?;
        let (project, _) = import_blend(&blender, root.path(), &format!("project_{case}"), &blend)?;
        let scene = read_scene(&project)?;
        let active = scene["active_scene"]
            .as_str()
            .ok_or("active scene is missing")?
            .to_owned();
        let applied = apply_operations(
            &project,
            root.path(),
            json!([
                {"op":"camera.create","id":"camera_main","transform":{"translation":[0.0,0.0,5.0]},"projection":"orthographic","ortho_scale":ORTHO_SCALE},
                {"op":"scene.update","target":{"id":active},"set":{"camera":"camera_main"}},
                {"op":"render.update","target":{"id":active},"set":{"resolution_x":SIZE,"resolution_y":SIZE,"resolution_percentage":100,"samples":1,"seed":0}}
            ]),
        )?;
        assert_eq!(applied["ok"], true, "{case}: {applied}");

        let frames = root.path().join(format!("frames_{case}"));
        let mut render = pot();
        render
            .arg("render")
            .arg(&project)
            .args(["--engine", "realtime", "--format", "exr", "--out"])
            .arg(&frames)
            .arg("--json");
        let output = checked(&mut render, "Potter realtime Thin Wall render")?;
        let response: Value = serde_json::from_slice(&output.stdout)?;
        if leaf {
            assert!(
                response["result"]["warnings"]
                    .as_array()
                    .is_some_and(|warnings| {
                        warnings.iter().any(|warning| {
                            warning["code"] == "UNSUPPORTED_FEATURE"
                                && warning["feature_id"] == "shader.principled.thin_wall"
                        })
                    }),
                "{case}: missing honest realtime Thin Wall subsurface warning: {response}"
            );
        } else {
            assert_eq!(
                response["result"]["warnings"],
                json!([]),
                "{case}: {response}"
            );
        }

        let exr = frames.join("frame_0001.exr");
        let loaded = apply_operations(
            &project,
            root.path(),
            json!([{"op":"image.load","id":"render","path":exr.to_string_lossy()}]),
        )?;
        assert_eq!(loaded["ok"], true, "{case}: {loaded}");
        let image: Image =
            serde_json::from_value(read_scene(&project)?["images"]["render"].clone())?;
        let data = load_image_data(&image, &project, ImageInterpolation::Closest)?;
        if leaf {
            assert_eq!(capture.len(), (SIZE * SIZE * 12) as usize, "{case}");
        } else {
            assert_matches_eevee(case, &data, capture)?;
        }
    }
    Ok(())
}

fn assert_matches_eevee(case: &str, data: &ImageData, bytes: &[u8]) -> TestResult {
    let pixel_count = (SIZE * SIZE) as usize;
    if bytes.len() != pixel_count * 12 {
        return Err(format!(
            "{case}: EEVEE capture has {} bytes, expected {}",
            bytes.len(),
            pixel_count * 12
        )
        .into());
    }
    let mut compared = 0;
    let mut worst = (0.0_f64, [0.0; 2], [0.0; 3], [0.0; 3]);
    for row in 0..SIZE {
        for column in 0..SIZE {
            let uv = [
                (f64::from(column) + 0.5) / f64::from(SIZE),
                (f64::from(row) + 0.5) / f64::from(SIZE),
            ];
            let pixel = (row * SIZE + column) as usize;
            let offset = pixel * 12;
            let wanted = [0, 1, 2].map(|channel| {
                let at = offset + channel * 4;
                f64::from(f32::from_le_bytes([
                    bytes[at],
                    bytes[at + 1],
                    bytes[at + 2],
                    bytes[at + 3],
                ]))
            });
            let sampled = sample(data, uv, false);
            let actual = [sampled[0], sampled[1], sampled[2]];
            let deviation = actual
                .iter()
                .zip(wanted)
                .map(|(actual, wanted)| (actual - wanted).abs())
                .fold(0.0_f64, f64::max);
            compared += 1;
            if deviation > worst.0 {
                worst = (deviation, uv, actual, wanted);
            }
        }
    }
    if compared < 100 {
        return Err(format!("{case}: only {compared} EEVEE pixels were compared").into());
    }
    assert!(
        worst.0 <= PIXEL_TOLERANCE,
        "{case}: Potter {:?} differs from EEVEE {:?} by {} at uv {:?} across {compared} pixels (tolerance {PIXEL_TOLERANCE})",
        worst.2,
        worst.3,
        worst.0,
        worst.1
    );
    Ok(())
}
