use std::{collections::BTreeSet, error::Error, fs, io::Cursor, path::Path, process::Output};

use assert_cmd::Command;
use serde_json::{Value, json};
use tempfile::tempdir;

const SEGMENTS: u32 = 24;
const RINGS: u32 = 12;

fn pot() -> Command {
    Command::new(env!("CARGO_BIN_EXE_pot"))
}

fn run_ok(command: &mut Command) -> Result<Output, Box<dyn Error>> {
    let output = command.output()?;
    if !output.status.success() {
        return Err(
            std::io::Error::other(String::from_utf8_lossy(&output.stdout).into_owned()).into(),
        );
    }
    Ok(output)
}

/// Builds a UV sphere scene; `smooth` adds `shade_smooth` to every face.
fn sphere_scene(scene: &Path, smooth: bool) -> Result<(), Box<dyn Error>> {
    run_ok(pot().arg("init").arg(scene).arg("--json"))?;
    let mut operations = vec![json!({
        "op": "node.create",
        "id": "ball",
        "kind": "sphere",
        "params": {"radius": 1.0, "segments": SEGMENTS, "ring_count": RINGS}
    })];
    if smooth {
        let face_ids: Vec<String> = (0..SEGMENTS * RINGS).map(|id| format!("f{id}")).collect();
        operations.push(json!({
            "op": "mesh.shade_smooth",
            "target": {"id": "ball"},
            "elements": {"domain": "face", "ids": face_ids}
        }));
    }
    let batch = json!({"schema_version": 1, "base_revision": 0, "operations": operations});
    let batch_file = scene.with_extension("json");
    fs::write(&batch_file, batch.to_string())?;
    run_ok(pot().arg("apply").arg(scene).arg("--file").arg(&batch_file))?;
    Ok(())
}

/// Renders the isometric normal pass and returns the distinct opaque RGB colours it contains.
fn normal_pass_colors(scene: &Path, out: &Path) -> Result<(Vec<u8>, usize), Box<dyn Error>> {
    run_ok(
        pot()
            .arg("preview")
            .arg(scene)
            .args([
                "--views", "iso", "--mode", "normal", "--size", "192", "--out",
            ])
            .arg(out)
            .args(["--overwrite", "--json"]),
    )?;
    let bytes = fs::read(out.join("iso.png"))?;
    let mut reader = png::Decoder::new(Cursor::new(bytes.as_slice())).read_info()?;
    let mut pixels = vec![
        0;
        reader
            .output_buffer_size()
            .ok_or("PNG output buffer size is unknown")?
    ];
    let info = reader.next_frame(&mut pixels)?;
    let channels = match info.color_type {
        png::ColorType::Rgba => 4,
        png::ColorType::Rgb => 3,
        _ => return Err("normal preview must be an RGB or RGBA PNG".into()),
    };
    let colors = pixels
        .chunks_exact(channels)
        .map(|pixel| [pixel[0], pixel[1], pixel[2]])
        .collect::<BTreeSet<_>>();
    Ok((pixels, colors.len()))
}

#[test]
fn smooth_sphere_normals_vary_within_faces_while_flat_sphere_stays_faceted()
-> Result<(), Box<dyn Error>> {
    let directory = tempdir()?;
    let flat_scene = directory.path().join("flat");
    let smooth_scene = directory.path().join("smooth");
    sphere_scene(&flat_scene, false)?;
    sphere_scene(&smooth_scene, true)?;

    let (flat_pixels, flat_colors) =
        normal_pass_colors(&flat_scene, &directory.path().join("flat-out"))?;
    let (smooth_pixels, smooth_colors) =
        normal_pass_colors(&smooth_scene, &directory.path().join("smooth-out"))?;

    assert_ne!(
        flat_pixels, smooth_pixels,
        "shade_smooth must change the preview"
    );
    assert!(
        smooth_colors > 2 * flat_colors,
        "smooth normals should take many more colours than {flat_colors} flat facets; got {smooth_colors}",
    );
    Ok(())
}

#[test]
fn flat_sphere_preview_has_no_warnings() -> Result<(), Box<dyn Error>> {
    let directory = tempdir()?;
    let scene = directory.path().join("flat");
    sphere_scene(&scene, false)?;
    let output = run_ok(
        pot()
            .arg("preview")
            .arg(&scene)
            .args(["--views", "iso", "--size", "64", "--out"])
            .arg(directory.path().join("out"))
            .args(["--overwrite", "--json"]),
    )?;
    let response: Value = serde_json::from_slice(&output.stdout)?;
    let warnings = &response["result"]["previews"][0]["warnings"];
    assert!(
        warnings.as_array().is_none_or(Vec::is_empty),
        "unexpected warnings: {warnings}"
    );
    Ok(())
}
