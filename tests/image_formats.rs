#![expect(
    clippy::expect_used,
    reason = "format fixture setup uses fixed valid images"
)]

use std::{error::Error, fs, io::BufReader, path::Path, process::Output};

use assert_cmd::Command;
use image::{DynamicImage, ImageBuffer, ImageFormat, Rgb, Rgba};
use proptest::prelude::*;
use serde_json::{Value, json};
use tempfile::tempdir;

fn pot() -> Command {
    Command::new(env!("CARGO_BIN_EXE_pot"))
}

fn init(scene: &Path) -> Result<(), Box<dyn Error>> {
    let output = pot().arg("init").arg(scene).arg("--json").output()?;
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stdout)
    );
    Ok(())
}

fn apply(scene: &Path, operations: &Value) -> Result<Output, Box<dyn Error>> {
    let batch = scene.with_extension("operations.json");
    fs::write(
        &batch,
        serde_json::to_vec(&json!({
            "schema_version": 1,
            "base_revision": 0,
            "operations": operations,
        }))?,
    )?;
    Ok(pot()
        .arg("apply")
        .arg(scene)
        .arg("--file")
        .arg(batch)
        .arg("--json")
        .output()?)
}

fn save_rgb(path: &Path, format: ImageFormat, pixel: [u8; 3]) -> Result<(), Box<dyn Error>> {
    let image = ImageBuffer::from_pixel(1, 1, Rgb(pixel));
    DynamicImage::ImageRgb8(image).save_with_format(path, format)?;
    Ok(())
}

fn save_png(path: &Path, pixel: [u8; 3]) -> Result<(), Box<dyn Error>> {
    let mut encoder = png::Encoder::new(fs::File::create(path)?, 1, 1);
    encoder.set_color(png::ColorType::Rgb);
    encoder.set_depth(png::BitDepth::Eight);
    encoder.write_header()?.write_image_data(&pixel)?;
    Ok(())
}

fn image_pixel(scene: &Path, id: &str) -> Result<[f64; 4], Box<dyn Error>> {
    let scene_doc: Value = serde_json::from_slice(&fs::read(scene.join("scene.json"))?)?;
    let image: potter::model::Image = serde_json::from_value(scene_doc["images"][id].clone())?;
    let data =
        potter::image::load_image_data(&image, scene, potter::image::ImageInterpolation::Closest)?;
    Ok(potter::image::sample(&data, [0.5, 0.5], 0))
}
fn assert_pixel(
    scene: &Path,
    id: &str,
    expected: [f64; 4],
    tolerance: f64,
) -> Result<(), Box<dyn Error>> {
    let actual = image_pixel(scene, id)?;
    for (actual, expected) in actual.into_iter().zip(expected) {
        assert!(
            (actual - expected).abs() <= tolerance,
            "{id}: actual pixel {actual:?} differs from expected {expected:?}"
        );
    }
    Ok(())
}

fn srgb_to_linear(value: f64) -> f64 {
    if value <= 0.04045 {
        value / 12.92
    } else {
        ((value + 0.055) / 1.055).powf(2.4)
    }
}

#[test]
fn image_load_decodes_supported_raster_formats_and_preserves_precision()
-> Result<(), Box<dyn Error>> {
    let directory = tempdir()?;
    let scene = directory.path().join("scene");
    init(&scene)?;
    let assets = directory.path().join("source");
    fs::create_dir_all(&assets)?;

    let png = assets.join("color.png");
    save_png(&png, [128, 64, 32])?;
    let jpeg = assets.join("color.jpg");
    save_rgb(&jpeg, ImageFormat::Jpeg, [128, 64, 32])?;
    let tiff8 = assets.join("color8.tiff");
    let tiff8_image = ImageBuffer::from_pixel(1, 1, Rgba([128_u8, 64, 32, 128]));
    DynamicImage::ImageRgba8(tiff8_image).save_with_format(&tiff8, ImageFormat::Tiff)?;
    let tiff16 = assets.join("color16.tiff");
    let tiff16_image = ImageBuffer::from_pixel(1, 1, Rgba([32_768_u16, 16_384, 65_535, 32_768]));
    DynamicImage::ImageRgba16(tiff16_image).save_with_format(&tiff16, ImageFormat::Tiff)?;
    let tiff_float = assets.join("float.tiff");
    let tiff_float_image = ImageBuffer::from_pixel(1, 1, Rgb([1.5_f32, 0.5_f32, 0.25_f32]));
    DynamicImage::ImageRgb32F(tiff_float_image).save_with_format(&tiff_float, ImageFormat::Tiff)?;
    let webp = assets.join("color.webp");
    save_rgb(&webp, ImageFormat::WebP, [128, 64, 32])?;
    let bmp = assets.join("color.bmp");
    save_rgb(&bmp, ImageFormat::Bmp, [128, 64, 32])?;
    let tga = assets.join("color.tga");
    save_rgb(&tga, ImageFormat::Tga, [128, 64, 32])?;
    let hdr = assets.join("float.hdr");
    let hdr_image = ImageBuffer::from_pixel(1, 1, Rgb([1.5_f32, 0.5_f32, 0.25_f32]));
    DynamicImage::ImageRgb32F(hdr_image).save_with_format(&hdr, ImageFormat::Hdr)?;

    let image_paths = [
        ("png_color", &png, None),
        ("jpeg_color", &jpeg, None),
        ("tiff8_color", &tiff8, None),
        ("tiff16_linear", &tiff16, Some("linear")),
        ("tiff_float_auto", &tiff_float, None),
        ("tiff_float_srgb", &tiff_float, Some("srgb")),
        ("webp_color", &webp, None),
        ("bmp_non_color", &bmp, Some("non_color")),
        ("tga_color", &tga, None),
        ("hdr_auto", &hdr, None),
    ];
    let operations: Vec<Value> = image_paths
        .iter()
        .map(|(id, path, colorspace)| {
            let mut operation = json!({
                "op": "image.load",
                "id": id,
                "path": path.to_string_lossy(),
            });
            if let Some(colorspace) = colorspace {
                operation["colorspace"] = json!(colorspace);
            }
            operation
        })
        .collect();
    let output = apply(&scene, &json!(operations))?;
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stdout)
    );

    let encoded = [128.0 / 255.0, 64.0 / 255.0, 32.0 / 255.0];
    let expected = encoded.map(srgb_to_linear);
    assert_pixel(
        &scene,
        "png_color",
        [expected[0], expected[1], expected[2], 1.0],
        1.0e-12,
    )?;
    assert_pixel(
        &scene,
        "jpeg_color",
        [expected[0], expected[1], expected[2], 1.0],
        0.04,
    )?;
    assert_pixel(
        &scene,
        "tiff8_color",
        [expected[0], expected[1], expected[2], 128.0 / 255.0],
        1.0e-12,
    )?;
    assert_pixel(
        &scene,
        "tiff16_linear",
        [
            32_768.0 / 65_535.0,
            16_384.0 / 65_535.0,
            1.0,
            32_768.0 / 65_535.0,
        ],
        1.0e-12,
    )?;
    assert_pixel(&scene, "tiff_float_auto", [1.5, 0.5, 0.25, 1.0], 1.0e-6)?;
    assert_pixel(
        &scene,
        "tiff_float_srgb",
        [
            srgb_to_linear(1.5),
            srgb_to_linear(0.5),
            srgb_to_linear(0.25),
            1.0,
        ],
        1.0e-6,
    )?;
    assert_pixel(
        &scene,
        "webp_color",
        [expected[0], expected[1], expected[2], 1.0],
        0.08,
    )?;
    assert_pixel(
        &scene,
        "bmp_non_color",
        [encoded[0], encoded[1], encoded[2], 1.0],
        1.0e-12,
    )?;
    assert_pixel(
        &scene,
        "tga_color",
        [expected[0], expected[1], expected[2], 1.0],
        1.0e-12,
    )?;
    assert_pixel(&scene, "hdr_auto", [1.5, 0.5, 0.25, 1.0], 0.01)?;
    Ok(())
}

#[test]
fn image_load_reports_corrupt_png_with_a_typed_error() -> Result<(), Box<dyn Error>> {
    let directory = tempdir()?;
    let scene = directory.path().join("scene");
    let source = directory.path().join("truncated.png");
    init(&scene)?;
    fs::write(&source, b"\x89PNG\r\n\x1a\ntruncated")?;
    let output = apply(
        &scene,
        &json!([{
            "op": "image.load",
            "id": "truncated",
            "path": source.to_string_lossy(),
        }]),
    )?;
    assert!(!output.status.success());
    let response: Value = serde_json::from_slice(&output.stdout)?;
    assert_eq!(response["error"]["code"], "INVALID_ARGUMENT");
    let feature_id = response["error"]["details"]["feature_id"]
        .as_str()
        .ok_or_else(|| std::io::Error::other("decode error lacks a feature ID"))?;
    assert_eq!(feature_id, "image.encoding.png.invalid_data");
    Ok(())
}

#[test]
fn jpeg_base_color_texture_appears_in_beauty_preview() -> Result<(), Box<dyn Error>> {
    let directory = tempdir()?;
    let scene = directory.path().join("scene");
    let source = directory.path().join("red.jpg");
    let output_dir = directory.path().join("preview");
    init(&scene)?;
    save_rgb(&source, ImageFormat::Jpeg, [255, 0, 0])?;
    let output = apply(
        &scene,
        &json!([
            {"op":"image.load","id":"red","path":source.to_string_lossy()},
            {"op":"material.create","id":"textured","base_color_texture":{"image":"red","interpolation":"closest"}},
            {"op":"node.create","id":"body","kind":"box","params":{"size":2.0},"transform":{"translation":[0.0,0.0,0.1],"scale":[1.0,1.0,0.1]},"material":"textured"},
            {"op":"light.create","id":"key","light_type":"point","energy":120.0,"transform":{"translation":[0.0,0.0,3.0]}}
        ]),
    )?;
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stdout)
    );
    let preview = pot()
        .arg("preview")
        .arg(&scene)
        .args([
            "--views", "top", "--size", "64", "--mode", "beauty", "--out",
        ])
        .arg(&output_dir)
        .arg("--json")
        .output()?;
    assert!(
        preview.status.success(),
        "{}",
        String::from_utf8_lossy(&preview.stdout)
    );
    let file = BufReader::new(fs::File::open(output_dir.join("top.png"))?);
    let mut decoder = png::Decoder::new(file);
    decoder.set_transformations(png::Transformations::EXPAND | png::Transformations::STRIP_16);
    let mut reader = decoder.read_info()?;
    let width = reader.info().width;
    let height = reader.info().height;
    let mut bytes = vec![
        0;
        reader
            .output_buffer_size()
            .expect("PNG output buffer size is known")
    ];
    let frame = reader.next_frame(&mut bytes)?;
    let center = (height as usize / 2 * width as usize + width as usize / 2)
        * match frame.color_type {
            png::ColorType::Grayscale => 1,
            png::ColorType::GrayscaleAlpha => 2,
            png::ColorType::Rgb => 3,
            png::ColorType::Rgba => 4,
            png::ColorType::Indexed => {
                return Err(std::io::Error::other("preview PNG palette was not expanded").into());
            }
        };
    let rgb = match frame.color_type {
        png::ColorType::Grayscale | png::ColorType::GrayscaleAlpha => {
            [bytes[center], bytes[center], bytes[center]]
        }
        png::ColorType::Rgb | png::ColorType::Rgba => {
            [bytes[center], bytes[center + 1], bytes[center + 2]]
        }
        png::ColorType::Indexed => {
            return Err(std::io::Error::other("preview PNG palette was not expanded").into());
        }
    };
    assert!(
        rgb[0] > rgb[1].saturating_add(20) && rgb[0] > rgb[2].saturating_add(20),
        "textured beauty pixel is not red: {rgb:?}"
    );
    Ok(())
}

#[test]
fn exr_image_load_unpremultiplies_alpha_and_keeps_linear_values() -> Result<(), Box<dyn Error>> {
    use exr::prelude::{
        Encoding, Image as ExrImage, ImageAttributes, IntegerBounds, Layer, LayerAttributes,
        SpecificChannels, Vec2, WritableImage,
    };
    let directory = tempdir()?;
    let scene = directory.path().join("scene");
    init(&scene)?;
    let source = directory.path().join("premultiplied.exr");
    let size = Vec2(1_usize, 1_usize);
    let auxiliary = Layer::new(
        size,
        LayerAttributes::named("Auxiliary"),
        Encoding::FAST_LOSSLESS,
        SpecificChannels::rgba(|_| (0.8_f32, 0.6_f32, 0.4_f32, 0.5_f32)),
    );
    let combined = Layer::new(
        size,
        LayerAttributes::named("Combined"),
        Encoding::FAST_LOSSLESS,
        SpecificChannels::rgba(|_| (0.25_f32, 0.1_f32, 0.05_f32, 0.5_f32)),
    );
    ExrImage::empty(ImageAttributes::new(IntegerBounds::from_dimensions(size)))
        .with_layer(auxiliary)
        .with_layer(combined)
        .write()
        .to_file(&source)?;
    let output = apply(
        &scene,
        &json!([{
            "op": "image.load",
            "id": "exr",
            "path": source.to_string_lossy(),
        }]),
    )?;
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stdout)
    );
    assert_pixel(&scene, "exr", [0.5, 0.2, 0.1, 0.5], 0.002)?;
    Ok(())
}
#[test]
fn half_float_exr_samples_decode_into_f64_pixels() -> Result<(), Box<dyn Error>> {
    let directory = tempdir()?;
    let scene = directory.path().join("scene");
    init(&scene)?;
    let source = directory.path().join("half.exr");
    exr::prelude::write_rgba_file(&source, 1, 1, |_, _| {
        (
            exr::prelude::f16::from_f32(0.25),
            exr::prelude::f16::from_f32(0.1),
            exr::prelude::f16::from_f32(0.05),
            exr::prelude::f16::from_f32(0.5),
        )
    })?;
    let output = apply(
        &scene,
        &json!([{
            "op": "image.load",
            "id": "half_exr",
            "path": source.to_string_lossy(),
        }]),
    )?;
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stdout)
    );
    assert_pixel(&scene, "half_exr", [0.5, 0.2, 0.1, 0.5], 0.002)?;
    Ok(())
}

fn blender_executable() -> Option<std::path::PathBuf> {
    fn usable(path: std::path::PathBuf) -> Option<std::path::PathBuf> {
        if !path.is_file() {
            return None;
        }
        let output = std::process::Command::new(&path)
            .arg("--version")
            .output()
            .ok()?;
        output.status.success().then_some(path)
    }

    if let Some(path) = std::env::var_os("POTTER_BLENDER") {
        return usable(std::path::PathBuf::from(path));
    }
    if let Some(paths) = std::env::var_os("PATH")
        && let Some(path) = std::env::split_paths(&paths)
            .map(|directory| directory.join("blender"))
            .find_map(usable)
    {
        return Some(path);
    }
    usable(std::path::PathBuf::from(
        "/Applications/Blender.app/Contents/MacOS/Blender",
    ))
}

fn contains_image_reference(value: &Value, image_id: &str) -> bool {
    match value {
        Value::Object(object) => object.iter().any(|(key, value)| {
            (key == "image" && value.as_str() == Some(image_id))
                || contains_image_reference(value, image_id)
        }),
        Value::Array(values) => values
            .iter()
            .any(|value| contains_image_reference(value, image_id)),
        _ => false,
    }
}
#[test]
fn blender_saved_jpeg_tiff_and_exr_match_blender_pixels() -> Result<(), Box<dyn Error>> {
    let Some(blender) = blender_executable() else {
        eprintln!("skipping Blender image parity test: Blender is unavailable");
        return Ok(());
    };
    let directory = tempdir()?;
    let root = directory.path();
    let scene = root.join("scene");
    init(&scene)?;
    let script = root.join("save_formats.py");
    fs::write(
        &script,
        r#"
import bpy
import json
import os
import sys
root = sys.argv[sys.argv.index("--") + 1]
pixel = [0.25, 0.5, 0.75, 1.0]
expected = {}
for expected_key, extension, file_format in (
    ("jpeg", "jpg", "JPEG"), ("tiff", "tif", "TIFF"), ("exr", "exr", "OPEN_EXR")
):
    image = bpy.data.images.new(
        "Pixel Grid " + expected_key, width=2, height=2, alpha=True, float_buffer=True
    )
    image.pixels[:] = pixel * 4
    path = os.path.join(root, "grid." + extension)
    image.file_format = file_format
    image.filepath_raw = path
    image.save()
    if not os.path.isfile(path):
        raise RuntimeError("Blender did not save " + path)
    decoded = bpy.data.images.load(path, check_existing=False)
    decoded.colorspace_settings.name = {
        "jpeg": "sRGB", "tiff": "Linear Rec.709", "exr": "Linear Rec.709"
    }[expected_key]
    expected[expected_key] = list(decoded.pixels[:4])
with open(os.path.join(root, "blender_pixels.json"), "w", encoding="utf-8") as output:
    json.dump(expected, output)
"#,
    )?;
    let fixture = std::process::Command::new(&blender)
        .args(["--background", "--factory-startup", "--python"])
        .arg(&script)
        .args([
            "--",
            root.to_str()
                .ok_or_else(|| std::io::Error::other("fixture directory path is not UTF-8"))?,
        ])
        .output()?;
    assert!(
        fixture.status.success(),
        "{}",
        String::from_utf8_lossy(&fixture.stderr)
    );
    let specifications = [
        ("blender_jpeg", "jpeg", "jpg", "srgb", 0.08),
        ("blender_tiff", "tiff", "tif", "linear", 0.002),
        ("blender_exr", "exr", "exr", "linear", 0.002),
    ];
    let operations: Vec<Value> = specifications
        .iter()
        .map(|(id, _, extension, colorspace, _)| {
            json!({
                "op": "image.load",
                "id": id,
                "path": root.join(format!("grid.{extension}")).to_string_lossy(),
                "colorspace": colorspace,
            })
        })
        .collect();
    let loaded = apply(&scene, &json!(operations))?;
    assert!(
        loaded.status.success(),
        "{}",
        String::from_utf8_lossy(&loaded.stdout)
    );
    let blender_pixels: Value =
        serde_json::from_slice(&fs::read(root.join("blender_pixels.json"))?)?;
    for (id, extension, _, colorspace, tolerance) in specifications {
        let pixel = blender_pixels[extension]
            .as_array()
            .ok_or_else(|| std::io::Error::other("Blender pixel sample is missing"))?;
        let mut expected = [
            pixel[0]
                .as_f64()
                .ok_or_else(|| std::io::Error::other("red is missing"))?,
            pixel[1]
                .as_f64()
                .ok_or_else(|| std::io::Error::other("green is missing"))?,
            pixel[2]
                .as_f64()
                .ok_or_else(|| std::io::Error::other("blue is missing"))?,
            pixel[3]
                .as_f64()
                .ok_or_else(|| std::io::Error::other("alpha is missing"))?,
        ];
        if colorspace == "srgb" {
            for component in &mut expected[..3] {
                *component = srgb_to_linear(*component);
            }
        }
        assert_pixel(&scene, id, expected, tolerance)?;
    }
    Ok(())
}

#[test]
fn blender_import_copies_and_resolves_exr_texture_images() -> Result<(), Box<dyn Error>> {
    let Some(blender) = blender_executable() else {
        eprintln!("skipping Blender EXR texture test: Blender is unavailable");
        return Ok(());
    };
    let directory = tempdir()?;
    let scene = directory.path().join("scene");
    init(&scene)?;
    let source = directory.path().join("texture.exr");
    exr::prelude::write_rgba_file(&source, 1, 1, |_, _| (0.4, 0.2, 0.1, 1.0))?;
    let blend_file = directory.path().join("textured.blend");
    let script = directory.path().join("create_blend.py");
    fs::write(
        &script,
        r#"
import bpy
import sys
source, target = sys.argv[sys.argv.index("--") + 1:]
bpy.ops.wm.read_factory_settings(use_empty=True)
image = bpy.data.images.load(source, check_existing=False)
material = bpy.data.materials.new("Textured")
material.use_nodes = True
material.use_fake_user = True
texture = material.node_tree.nodes.new("ShaderNodeTexImage")
texture.image = image
shader = next(node for node in material.node_tree.nodes if node.type == "BSDF_PRINCIPLED")
material.node_tree.links.new(texture.outputs["Color"], shader.inputs["Base Color"])
bpy.ops.wm.save_as_mainfile(filepath=target)
"#,
    )?;
    let fixture = std::process::Command::new(&blender)
        .args(["--background", "--factory-startup", "--python"])
        .arg(&script)
        .args([
            "--",
            source
                .to_str()
                .ok_or_else(|| std::io::Error::other("EXR path is not UTF-8"))?,
            blend_file
                .to_str()
                .ok_or_else(|| std::io::Error::other("blend path is not UTF-8"))?,
        ])
        .output()?;
    assert!(
        fixture.status.success(),
        "{}",
        String::from_utf8_lossy(&fixture.stderr)
    );
    let imported = pot()
        .args(["import"])
        .arg(&scene)
        .args(["--file"])
        .arg(&blend_file)
        .args([
            "--format",
            "blend",
            "--mode",
            "replace",
            "--base-revision",
            "0",
            "--blender",
        ])
        .arg(&blender)
        .arg("--json")
        .output()?;
    assert!(
        imported.status.success(),
        "{}",
        String::from_utf8_lossy(&imported.stdout)
    );
    let document: Value = serde_json::from_slice(&fs::read(scene.join("scene.json"))?)?;
    let images = document["images"]
        .as_object()
        .ok_or_else(|| std::io::Error::other("imported image registry is missing"))?;
    let (image_id, _) = images
        .iter()
        .next()
        .ok_or_else(|| std::io::Error::other("EXR texture was not imported"))?;
    assert_pixel(&scene, image_id, [0.4, 0.2, 0.1, 1.0], 0.002)?;
    let material = document["materials"]
        .as_object()
        .and_then(|materials| {
            materials
                .values()
                .find(|material| material["name"] == "Textured")
        })
        .ok_or_else(|| std::io::Error::other("textured Blender material was not imported"))?;
    let node_tree_id = material["node_tree"]
        .as_str()
        .ok_or_else(|| std::io::Error::other("shader node tree was not imported"))?;
    assert!(
        contains_image_reference(&document["node_groups"][node_tree_id], image_id),
        "image texture node does not reference its imported image ID"
    );
    Ok(())
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(32))]

    #[test]
    fn truncated_png_returns_an_error_without_panicking(cut in 0_usize..40) {
        let mut bytes = Vec::new();
        {
            let mut encoder = png::Encoder::new(&mut bytes, 1, 1);
            encoder.set_color(png::ColorType::Rgb);
            encoder.set_depth(png::BitDepth::Eight);
            encoder.write_header().expect("PNG writer starts").write_image_data(&[32, 64, 128])
                .expect("fixture PNG can be written");
        }
        prop_assume!(cut < bytes.len());
        let result = std::panic::catch_unwind(|| {
            potter::image::decode_pixels(&bytes[..cut], potter::model::ImageColorspace::Srgb)
        });
        let decoded = result.expect("decoder did not panic");
        let Err(error) = decoded else {
            return Err(proptest::test_runner::TestCaseError::fail("truncated input decoded successfully"));
        };
        prop_assert!(error.details.get("feature_id").is_some());
    }
}
