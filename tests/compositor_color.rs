use std::{error::Error, fs, io::BufReader, path::Path};

use assert_cmd::Command;

use potter_core::{
    color::{ColorManagement, ViewTransform, linear_to_srgb},
    compositor::{BlendMode, Image, gaussian_blur, mix_rgb},
};
use proptest::prelude::*;
use tempfile::tempdir;

fn pot() -> Command {
    Command::new(env!("CARGO_BIN_EXE_pot"))
}

fn apply(scene: &Path, operations: &str) -> Result<(), Box<dyn Error>> {
    let operation_file = scene.with_extension("operations.json");
    fs::write(&operation_file, operations)?;
    let output = pot()
        .arg("apply")
        .arg(scene)
        .arg("--file")
        .arg(&operation_file)
        .arg("--json")
        .output()?;
    if !output.status.success() {
        return Err(
            std::io::Error::other(String::from_utf8_lossy(&output.stdout).into_owned()).into(),
        );
    }
    Ok(())
}

fn render(scene: &Path, output: &Path) -> Result<Vec<u8>, Box<dyn Error>> {
    let result = pot()
        .arg("render")
        .arg(scene)
        .args(["--format", "png", "--out"])
        .arg(output)
        .arg("--json")
        .output()?;
    if !result.status.success() {
        return Err(
            std::io::Error::other(String::from_utf8_lossy(&result.stdout).into_owned()).into(),
        );
    }
    let decoder = png::Decoder::new(BufReader::new(fs::File::open(
        output.join("frame_0001.png"),
    )?));
    let mut reader = decoder.read_info()?;
    let mut pixels = vec![
        0;
        reader
            .output_buffer_size()
            .ok_or("PNG output buffer size is unknown")?
    ];
    reader.next_frame(&mut pixels)?;
    Ok(pixels)
}

fn center_rgb(image: &[u8], width: usize, height: usize) -> [u8; 3] {
    let index = ((height / 2) * width + width / 2) * 4;
    [image[index], image[index + 1], image[index + 2]]
}

#[test]
fn cli_render_applies_compositor_and_scene_color_management() -> Result<(), Box<dyn Error>> {
    let directory = tempdir()?;
    let scene = directory.path().join("scene");
    let initialized = pot().arg("init").arg(&scene).arg("--json").output()?;
    assert!(initialized.status.success());
    apply(
        &scene,
        r#"{"schema_version":1,"base_revision":0,"operations":[
            {"op":"node.create","id":"body","kind":"box","params":{"size":1},"transform":{"translation":[0,0,0.5]}},
            {"op":"camera.create","id":"camera_main","transform":{"translation":[0,0,4]},"projection":"orthographic","ortho_scale":3},
            {"op":"light.create","id":"key_light","light_type":"point","transform":{"translation":[0,0,3]},"color":[1.0,0.1,0.1],"energy":1.0},
            {"op":"scene.update","target":{"id":"scene_main"},"set":{"camera":"camera_main"}},
            {"op":"render.update","target":{"id":"scene_main"},"set":{"resolution_x":32,"resolution_y":32,"samples":1}}
        ]}"#,
    )?;
    let baseline = render(&scene, &directory.path().join("baseline"))?;

    apply(
        &scene,
        r#"{"schema_version":1,"base_revision":1,"operations":[
            {"op":"graph.create","id":"graph_comp","kind":"compositor"},
            {"op":"compositor.node_add","graph":"graph_comp","id":"layers","type":"CompositorNodeRLayers"},
            {"op":"compositor.node_add","graph":"graph_comp","id":"gain","type":"CompositorNodeExposure","properties":{"exposure":0.5}},
            {"op":"compositor.node_add","graph":"graph_comp","id":"composite","type":"CompositorNodeComposite"},
            {"op":"compositor.link","graph":"graph_comp","from_node":"layers","from_socket":"Image","to_node":"gain","to_socket":"Image"},
            {"op":"compositor.link","graph":"graph_comp","from_node":"gain","from_socket":"Image","to_node":"composite","to_socket":"Image"},
            {"op":"compositor.enable","target":{"id":"scene_main"},"set":{"enabled":true,"graph":"graph_comp"}},
            {"op":"color.update","target":{"id":"scene_main"},"set":{"view_transform":"agx","exposure":0.5}}
        ]}"#,
    )?;
    let processed = render(&scene, &directory.path().join("processed"))?;
    let before = center_rgb(&baseline, 32, 32);
    let after = center_rgb(&processed, 32, 32);
    assert!(
        after[0] > before[0],
        "rendered red channel should brighten: {before:?} -> {after:?}"
    );
    assert_ne!(before, after);
    Ok(())
}

#[test]
fn cli_render_writes_requested_passes_as_multilayer_exr() -> Result<(), Box<dyn Error>> {
    let directory = tempdir()?;
    let scene = directory.path().join("scene");
    let initialized = pot().arg("init").arg(&scene).arg("--json").output()?;
    assert!(initialized.status.success());
    apply(
        &scene,
        r#"{"schema_version":1,"base_revision":0,"operations":[
            {"op":"node.create","id":"body","kind":"box","params":{"size":1},"transform":{"translation":[0,0,0.5]}},
            {"op":"camera.create","id":"camera_main","transform":{"translation":[0,0,4]},"projection":"orthographic","ortho_scale":3},
            {"op":"scene.update","target":{"id":"scene_main"},"set":{"camera":"camera_main"}},
            {"op":"render.update","target":{"id":"scene_main"},"set":{"resolution_x":16,"resolution_y":16,"samples":1}},
            {"op":"render.passes_update","target":{"id":"scene_main"},"set":{"passes":["combined","z","normal","albedo","emission","object_index","cryptomatte","ao"]}}
        ]}"#,
    )?;
    let output = directory.path().join("multilayer-exr");
    let rendered = pot()
        .arg("render")
        .arg(&scene)
        .args(["--format", "exr", "--out"])
        .arg(&output)
        .arg("--json")
        .output()?;
    assert!(
        rendered.status.success(),
        "{}",
        String::from_utf8_lossy(&rendered.stdout)
    );
    let bytes = fs::read(output.join("frame_0001.exr"))?;
    assert_eq!(bytes.get(0..4), Some(&[0x76, 0x2f, 0x31, 0x01][..]));
    for layer in [
        "Combined",
        "Depth",
        "Normal",
        "Albedo",
        "Emission",
        "ObjectIndex",
        "Cryptomatte",
        "AO",
    ] {
        assert!(
            bytes
                .windows(layer.len())
                .any(|window| window == layer.as_bytes()),
            "EXR should include the `{layer}` layer",
        );
    }
    Ok(())
}

#[test]
fn compositor_mask_node_rasterizes_scene_mask_data() -> Result<(), Box<dyn Error>> {
    let directory = tempdir()?;
    let scene = directory.path().join("scene");
    let initialized = pot().arg("init").arg(&scene).arg("--json").output()?;
    assert!(initialized.status.success());
    apply(
        &scene,
        r#"{"schema_version":1,"base_revision":0,"operations":[
            {"op":"camera.create","id":"camera_main","transform":{"translation":[0,0,4]},"projection":"orthographic","ortho_scale":3},
            {"op":"scene.update","target":{"id":"scene_main"},"set":{"camera":"camera_main"}},
            {"op":"render.update","target":{"id":"scene_main"},"set":{"resolution_x":16,"resolution_y":16,"samples":1}},
            {"op":"mask.create","id":"mask_main","splines":[{"feather":0.0,"points":[
                {"position":[2.0,2.0],"keyframes":[]},
                {"position":[14.0,2.0],"keyframes":[]},
                {"position":[14.0,14.0],"keyframes":[]},
                {"position":[2.0,14.0],"keyframes":[]}
            ]}]},
            {"op":"graph.create","id":"graph_comp","kind":"compositor"},
            {"op":"compositor.node_add","graph":"graph_comp","id":"mask","type":"CompositorNodeMask","properties":{"mask_id":"mask_main"}},
            {"op":"compositor.node_add","graph":"graph_comp","id":"composite","type":"CompositorNodeComposite"},
            {"op":"compositor.link","graph":"graph_comp","from_node":"mask","from_socket":"Mask","to_node":"composite","to_socket":"Image"},
            {"op":"compositor.enable","target":{"id":"scene_main"},"set":{"enabled":true,"graph":"graph_comp"}}
        ]}"#,
    )?;
    let pixels = render(&scene, &directory.path().join("mask-render"))?;
    assert_eq!(center_rgb(&pixels, 16, 16), [255, 255, 255]);
    assert_eq!(center_rgb_at(&pixels, 0, 0, 16), [0, 0, 0]);
    Ok(())
}

fn center_rgb_at(image: &[u8], x: usize, y: usize, width: usize) -> [u8; 3] {
    let index = (y * width + x) * 4;
    [image[index], image[index + 1], image[index + 2]]
}

#[test]
fn exposure_plus_one_doubles_raw_linear_values() {
    let settings = ColorManagement {
        view_transform: ViewTransform::Raw,
        exposure: 1.0,
        ..ColorManagement::default()
    };
    let result = settings.transform_linear([0.25, 0.5, 0.75]);
    assert_eq!(result, [0.5, 1.0, 1.5]);
}

#[test]
fn view_transforms_look_gamma_and_srgb_encoding_are_applied() {
    let sample = [0.25; 3];
    let standard = ColorManagement::default().transform_linear(sample);
    assert_eq!(standard, sample);

    let filmic = ColorManagement {
        view_transform: ViewTransform::Filmic,
        ..ColorManagement::default()
    }
    .transform_linear(sample);
    assert!(filmic[0] < standard[0]);

    let false_color = ColorManagement {
        view_transform: ViewTransform::FalseColor,
        ..ColorManagement::default()
    }
    .transform_linear(sample);
    assert_ne!(false_color[0], false_color[2]);

    let gamma = ColorManagement {
        view_transform: ViewTransform::Raw,
        gamma: 2.0,
        ..ColorManagement::default()
    }
    .transform_linear(sample);
    assert_eq!(gamma, [0.5; 3]);

    let look = ColorManagement {
        view_transform: ViewTransform::Raw,
        look: "high_contrast".to_owned(),
        ..ColorManagement::default()
    }
    .transform_linear([0.75; 3]);
    assert!(look[0] > 0.75);
    assert!((linear_to_srgb(0.5) - 0.735_356_983_052_449_5).abs() < 1.0e-12);
}

proptest! {
    #[test]
    fn agx_is_zero_preserving_and_monotonic(value in 0.0_f64..100.0, delta in 0.0_f64..10.0) {
        let settings = ColorManagement {
            view_transform: ViewTransform::AgX,
            ..ColorManagement::default()
        };
        let zero = settings.transform_linear([0.0; 3]);
        let lower = settings.transform_linear([value; 3]);
        let upper = settings.transform_linear([value + delta; 3]);
        prop_assert!(zero.iter().all(|channel| channel.abs() < 1.0e-12));
        for (low, high) in lower.into_iter().zip(upper) {
            prop_assert!(high + 1.0e-12 >= low);
        }
    }
}

#[test]
fn add_mix_sums_image_channels() {
    assert_eq!(
        mix_rgb(BlendMode::Add, [0.2, 0.3, 0.4, 0.5], [0.1, 0.2, 0.3, 0.4]),
        [0.3, 0.5, 0.700_000_05, 0.9],
    );
}

#[test]
fn gaussian_blur_preserves_mean_with_periodic_boundaries() -> Result<(), Box<dyn std::error::Error>>
{
    let pixels = (0..25)
        .map(|index| {
            let value = (index as f32) / 24.0;
            [value, value * 0.5, 1.0 - value, 1.0]
        })
        .collect::<Vec<_>>();
    let image = Image::new(5, 5, pixels)?;
    let blurred = gaussian_blur(&image, 1)?;
    for channel in 0..4 {
        let before = image.pixels.iter().map(|pixel| pixel[channel]).sum::<f32>() / 25.0;
        let after = blurred
            .pixels
            .iter()
            .map(|pixel| pixel[channel])
            .sum::<f32>()
            / 25.0;
        assert!((before - after).abs() < 1.0e-6);
    }
    Ok(())
}
