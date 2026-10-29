//! Realtime material displacement uses EEVEE shader math, while a path render uses Cycles math.
//!
//! `fixtures/native_oracle/displacement_math_semantics.py` makes a UV map with U > 1 feed an
//! ARCSINE node and then a material Displacement node. Its Blender 5.2.2 renders establish a
//! consumer-visible difference in the plane's image centroid between EEVEE and Cycles.

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

use std::{fs, io::BufReader, path::Path};

use native_oracle::{
    TestResult, checked, import_blend, node_id, pot, read_scene, run_blender_arguments,
    write_script,
};
use serde_json::Value;
use tempfile::tempdir;

const BLENDER_SCRIPT: &str = include_str!("fixtures/native_oracle/displacement_math_semantics.py");

fn visible_centroid_bottom_up(path: &Path) -> TestResult<f64> {
    let mut reader = png::Decoder::new(BufReader::new(fs::File::open(path)?)).read_info()?;
    let mut pixels = vec![
        0;
        reader
            .output_buffer_size()
            .ok_or("PNG output buffer size is unknown")?
    ];
    let info = reader.next_frame(&mut pixels)?;
    let channels = info.color_type.samples();
    if channels < 3 {
        return Err("displacement render is not an RGB image".into());
    }
    let width = info.width as usize;
    let height = info.height as usize;
    let mut row_sum = 0.0;
    let mut count = 0usize;
    for (index, pixel) in pixels[..info.buffer_size()]
        .chunks_exact(channels)
        .enumerate()
    {
        if pixel[0].max(pixel[1]).max(pixel[2]) > 64 {
            let row = index / width;
            row_sum += (height - row) as f64 - 0.5;
            count += 1;
        }
    }
    if count == 0 {
        return Err("displacement render contains no visible plane".into());
    }
    Ok(row_sum / count as f64 / height as f64)
}

#[test]
fn realtime_displacement_uses_eevee_and_path_displacement_uses_cycles_math() -> TestResult {
    let Some(blender) = blender_checked::blender_executable() else {
        eprintln!("skipping material displacement semantics oracle: Blender is unavailable");
        return Ok(());
    };
    let root = tempdir()?;
    let script = write_script(
        root.path(),
        "displacement_math_semantics.py",
        BLENDER_SCRIPT,
    )?;
    let blend = root.path().join("displacement.blend");
    let expected_json = root.path().join("expected.json");
    run_blender_arguments(
        &blender,
        &script,
        None,
        &[blend.as_path(), expected_json.as_path()],
    )?;
    let oracle: Value = serde_json::from_slice(&fs::read(&expected_json)?)?;
    assert_eq!(oracle["blender_version"], "5.2.2 LTS");
    let eevee_centroid = oracle["renders"]["eevee"]["centroid_bottom_up"]
        .as_f64()
        .ok_or("Blender EEVEE displacement centroid is missing")?;
    let cycles_centroid = oracle["renders"]["cycles"]["centroid_bottom_up"]
        .as_f64()
        .ok_or("Blender Cycles displacement centroid is missing")?;
    let oracle_separation = cycles_centroid - eevee_centroid;
    assert!(
        oracle_separation > 0.01,
        "Blender's EEVEE and Cycles displacement probes do not discriminate: EEVEE={eevee_centroid}, Cycles={cycles_centroid}"
    );

    let (project, _) = import_blend(&blender, root.path(), "project", &blend)?;
    let scene = read_scene(&project)?;
    let camera = node_id(&scene, "DisplacementCamera")?;
    let render = |engine: &str| -> TestResult<_> {
        let output = root.path().join(format!("potter_{engine}"));
        let mut command = pot();
        command
            .arg("render")
            .arg(&project)
            .args([
                "--camera",
                camera.as_str(),
                "--engine",
                engine,
                "--format",
                "png",
                "--out",
            ])
            .arg(&output)
            .arg("--json");
        checked(&mut command, "material displacement render")?;
        Ok(output.join("frame_0001.png"))
    };
    let realtime = visible_centroid_bottom_up(&render("realtime")?)?;
    let path = visible_centroid_bottom_up(&render("path")?)?;
    let potter_separation = path - realtime;
    assert!(
        potter_separation > oracle_separation * 0.5,
        "Potter's rendered displacement centroid separates by {potter_separation:.4}; Blender 5.2.2 EEVEE and Cycles separate by {oracle_separation:.4} (realtime={realtime:.4}, path={path:.4})"
    );
    Ok(())
}
