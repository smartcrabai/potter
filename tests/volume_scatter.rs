use std::{error::Error, f64::consts::FRAC_1_SQRT_2, fs, io::Cursor, path::Path, process::Command};

use serde_json::{Value, json};
use tempfile::tempdir;

fn pot(args: &[&str]) -> Result<Value, Box<dyn Error>> {
    let output = Command::new(env!("CARGO_BIN_EXE_pot"))
        .args(args)
        .arg("--json")
        .output()?;
    if !output.status.success() {
        return Err(std::io::Error::other(format!(
            "pot {:?} failed: {}{}",
            args,
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        ))
        .into());
    }
    Ok(serde_json::from_slice(&output.stdout)?)
}

fn png_pixels(path: &Path) -> Result<(u32, u32, Vec<u8>), Box<dyn Error>> {
    let mut reader = png::Decoder::new(Cursor::new(fs::read(path)?)).read_info()?;
    let mut pixels = vec![
        0;
        reader
            .output_buffer_size()
            .ok_or("PNG output buffer size overflow")?
    ];
    let info = reader.next_frame(&mut pixels)?;
    pixels.truncate(info.buffer_size());
    if info.color_type != png::ColorType::Rgba {
        return Err("rendered volume PNG is not RGBA".into());
    }
    Ok((info.width, info.height, pixels))
}

fn find_png(directory: &Path) -> Result<std::path::PathBuf, Box<dyn Error>> {
    fs::read_dir(directory)?
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .find(|path| path.extension().is_some_and(|extension| extension == "png"))
        .ok_or_else(|| "render did not create a PNG".into())
}

fn luminance(pixels: &[u8], width: u32, x: u32, y: u32) -> Result<f64, Box<dyn Error>> {
    let index = usize::try_from(y)?
        .checked_mul(usize::try_from(width)?)
        .and_then(|row| row.checked_add(usize::try_from(x).ok()?))
        .and_then(|pixel| pixel.checked_mul(4))
        .ok_or("pixel offset overflows usize")?;
    let rgb = pixels
        .get(index..index + 3)
        .ok_or("pixel is outside the image")?;
    Ok(0.2126 * f64::from(rgb[0]) + 0.7152 * f64::from(rgb[1]) + 0.0722 * f64::from(rgb[2]))
}

fn make_scattering_scene(root: &Path) -> Result<std::path::PathBuf, Box<dyn Error>> {
    let scene = root.join("scene");
    pot(&["init", scene.to_str().ok_or("scene path is not UTF-8")?])?;
    let mut values = Vec::with_capacity(8 * 8 * 8);
    for z in 0..8 {
        for y in 0..8 {
            for x in 0..8 {
                let dx = f64::from(x) - 3.5;
                let dy = f64::from(y) - 3.5;
                let dz = f64::from(z) - 3.5;
                values.push(if dx * dx + dy * dy + dz * dz < 12.25 {
                    1.0
                } else {
                    0.0
                });
            }
        }
    }
    let operations = json!({
        "schema_version": 1,
        "base_revision": 0,
        "operations": [
            {"op":"graph.create","id":"cloud_graph","kind":"shader"},
            {"op":"graph.node_add","graph":"cloud_graph","id":"output","type":"ShaderNodeOutputMaterial"},
            {"op":"graph.node_add","graph":"cloud_graph","id":"volume","type":"ShaderNodeVolumePrincipled","inputs":{"Density":1.0,"Color":[0.5,0.5,0.5,1.0],"Anisotropy":0.0}},
            {"op":"graph.link","graph":"cloud_graph","from_node":"volume","from_socket":"Volume","to_node":"output","to_socket":"Volume"},
            {"op":"material.create","id":"cloud_material","graph":"cloud_graph"},
            {"op":"volume.create","id":"cloud","materials":["cloud_material"],"grids":[{"dims":[8,8,8],"voxel_size":0.5,"origin":[-1.75,-1.75,-1.75],"values":values}]},
            {"op":"world.create","id":"world_main","color":[0.02,0.02,0.02],"strength":1.0},
            {"op":"camera.create","id":"camera_main","transform":{"translation":[0.0,0.0,6.0]},"projection":"orthographic","ortho_scale":5.0},
            {"op":"light.create","id":"sun_main","light_type":"sun","energy":20.0,"transform":{"rotation":[0.0,FRAC_1_SQRT_2,0.0,FRAC_1_SQRT_2]}},
            {"op":"scene.update","target":{"id":"scene_main"},"set":{"camera":"camera_main","world":"world_main"}},
            {"op":"render.update","target":{"id":"scene_main"},"set":{"resolution_x":32,"resolution_y":32,"samples":32,"seed":712,"max_bounces":4}}
        ]
    });
    let operations_path = root.join("operations.json");
    fs::write(&operations_path, serde_json::to_vec(&operations)?)?;
    pot(&[
        "apply",
        scene.to_str().ok_or("scene path is not UTF-8")?,
        "--file",
        operations_path
            .to_str()
            .ok_or("operations path is not UTF-8")?,
    ])?;
    Ok(scene)
}

#[test]
fn scattering_cloud_is_lit_in_beauty_and_path_renders_are_seed_deterministic()
-> Result<(), Box<dyn Error>> {
    let directory = tempdir()?;
    let scene = make_scattering_scene(directory.path())?;
    let preview_dir = directory.path().join("preview");
    pot(&[
        "preview",
        scene.to_str().ok_or("scene path is not UTF-8")?,
        "--camera",
        "camera_main",
        "--mode",
        "beauty",
        "--size",
        "64",
        "--out",
        preview_dir.to_str().ok_or("preview path is not UTF-8")?,
    ])?;
    let (width, height, preview) = png_pixels(&find_png(&preview_dir)?)?;
    let center_x = width / 2;
    let center_y = height / 2;
    let center = luminance(&preview, width, center_x, center_y)?;
    let lit_side = luminance(&preview, width, center_x + 4, center_y)?;
    let shadow_side = luminance(&preview, width, center_x - 4, center_y)?;
    let corner = luminance(&preview, width, 0, 0)?;
    assert!(
        center > corner + 2.0,
        "volume center {center} should differ from background {corner}"
    );
    assert!(
        lit_side > shadow_side + 1.0,
        "sun-lit cloud side {lit_side} should be brighter than shadow side {shadow_side}"
    );
    assert!(
        corner > 0.0,
        "background corners should retain the world environment"
    );

    let first_dir = directory.path().join("path_first");
    let second_dir = directory.path().join("path_second");
    for output in [&first_dir, &second_dir] {
        pot(&[
            "render",
            scene.to_str().ok_or("scene path is not UTF-8")?,
            "--engine",
            "path",
            "--format",
            "png",
            "--out",
            output.to_str().ok_or("render path is not UTF-8")?,
        ])?;
    }
    let first_path = first_dir.join("frame_0001.png");
    let second_path = second_dir.join("frame_0001.png");
    let path_pixels = fs::read(&first_path)?;
    assert_eq!(
        path_pixels,
        fs::read(&second_path)?,
        "path render must be deterministic for a fixed seed"
    );
    let (path_width, path_height, path_image) = png_pixels(&first_path)?;
    let path_center = luminance(&path_image, path_width, path_width / 2, path_height / 2)?;
    let path_lit = luminance(&path_image, path_width, path_width / 2 + 4, path_height / 2)?;
    let path_shadow = luminance(&path_image, path_width, path_width / 2 - 4, path_height / 2)?;
    let path_corner = luminance(&path_image, path_width, 0, 0)?;
    assert!(
        path_center > path_corner + 2.0,
        "path volume center {path_center} should be lit above background {path_corner}"
    );
    assert!(
        path_lit > path_shadow,
        "path sun-lit side {path_lit} should be brighter than shadow side {path_shadow}"
    );
    Ok(())
}
