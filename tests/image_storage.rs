#![expect(
    clippy::unwrap_used,
    reason = "image storage tests use fixed valid fixtures"
)]

use std::{collections::BTreeMap, error::Error, fs, path::Path};

use potter_core::{
    error::ErrorCode,
    hash,
    image::{
        ImageInterpolation, decode_png_dimensions, encode_pixels, load_image_data,
        load_image_data_with_staged,
    },
    model::{Image, ImageAlphaMode, ImageColorspace, ImageSource, ImageTile},
};
use tempfile::tempdir;

fn image(blob: Option<String>) -> Image {
    Image {
        name: "stored image".to_owned(),
        source: ImageSource::Packed,
        colorspace: ImageColorspace::Linear,
        width: 1,
        height: 1,
        tiles: Vec::new(),
        blob,
        source_path: None,
        source_hash: None,
        alpha_mode: ImageAlphaMode::Straight,
    }
}

fn write_blob(root: &Path, bytes: &[u8]) -> Result<String, Box<dyn Error>> {
    let reference = hash::sha256(bytes);
    let hex = reference
        .strip_prefix("sha256:")
        .ok_or_else(|| std::io::Error::other("missing hash prefix"))?;
    let path = root.join("assets").join("sha256").join(hex).join("blob");
    fs::create_dir_all(
        path.parent()
            .ok_or_else(|| std::io::Error::other("blob has no parent directory"))?,
    )?;
    fs::write(path, bytes)?;
    Ok(reference)
}

fn png_bytes(color: [u8; 4]) -> Result<Vec<u8>, Box<dyn Error>> {
    let mut bytes = Vec::new();
    {
        let mut encoder = png::Encoder::new(&mut bytes, 1, 1);
        encoder.set_color(png::ColorType::Rgba);
        encoder.set_depth(png::BitDepth::Eight);
        let mut writer = encoder.write_header()?;
        writer.write_image_data(&color)?;
    }
    Ok(bytes)
}

#[test]
fn packed_staged_and_udim_blobs_load_by_verified_content_hash() -> Result<(), Box<dyn Error>> {
    let directory = tempdir()?;
    let root = directory.path();
    let base_pixels = [[0.25, 0.5, 0.75, 1.0]];
    let base_bytes = encode_pixels(1, 1, &base_pixels)?;
    let base_blob = write_blob(root, &base_bytes)?;
    let tile_pixels = [[1.0, 0.0, 0.5, 1.0]];
    let tile_bytes = encode_pixels(1, 1, &tile_pixels)?;
    let tile_blob = write_blob(root, &tile_bytes)?;

    let mut packed = image(Some(base_blob.clone()));
    packed.tiles.push(ImageTile {
        number: 1012,
        width: 1,
        height: 1,
        blob: tile_blob,
    });
    let loaded = load_image_data(&packed, root, ImageInterpolation::Linear)?;
    assert_eq!(loaded.pixels, base_pixels);
    assert_eq!(loaded.tiles[&1012].pixels, tile_pixels);

    let staged = load_image_data_with_staged(
        &image(Some(base_blob.clone())),
        root,
        ImageInterpolation::Closest,
        &BTreeMap::from([(base_blob.clone(), base_bytes.clone())]),
    )?;
    assert_eq!(staged.pixels, base_pixels);
    assert_eq!(staged.interpolation, ImageInterpolation::Closest);

    let wrong_bytes = encode_pixels(1, 1, &[[0.9, 0.8, 0.7, 1.0]])?;
    let mismatched = load_image_data_with_staged(
        &image(Some(base_blob.clone())),
        root,
        ImageInterpolation::Linear,
        &BTreeMap::from([(base_blob, wrong_bytes.clone())]),
    );
    assert_eq!(mismatched.unwrap_err().code, ErrorCode::ValidationFailed);

    let missing = load_image_data(
        &image(Some(hash::sha256(&wrong_bytes))),
        root,
        ImageInterpolation::Linear,
    );
    assert_eq!(missing.unwrap_err().code, ErrorCode::FileNotFound);
    Ok(())
}

#[test]
fn linked_images_detect_changed_and_missing_sources() -> Result<(), Box<dyn Error>> {
    let directory = tempdir()?;
    let root = directory.path();
    let source_path = root.join("linked.rgba");
    let source_bytes = png_bytes([128, 64, 255, 64])?;
    assert_eq!(decode_png_dimensions(&source_bytes)?, (1, 1));
    fs::write(&source_path, &source_bytes)?;
    let mut linked = image(None);
    linked.source = ImageSource::File;
    linked.source_path = Some("linked.rgba".to_owned());
    linked.source_hash = Some(hash::sha256(&source_bytes));
    linked.colorspace = ImageColorspace::Srgb;

    let decoded = load_image_data(&linked, root, ImageInterpolation::Linear)?;
    assert!((decoded.pixels[0][0] - 0.215_860_500_113_899_26).abs() < 1.0e-12);
    assert!((decoded.pixels[0][1] - 0.051_269_458_374_043_24).abs() < 1.0e-12);
    assert!((decoded.pixels[0][2] - 1.0).abs() < 1.0e-12);
    assert!((decoded.pixels[0][3] - 64.0 / 255.0).abs() < 1.0e-12);
    fs::write(&source_path, encode_pixels(1, 1, &[[0.9, 0.8, 0.7, 1.0]])?)?;
    assert_eq!(
        load_image_data(&linked, root, ImageInterpolation::Linear)
            .unwrap_err()
            .code,
        ErrorCode::AssetChanged
    );
    fs::remove_file(&source_path)?;
    assert_eq!(
        load_image_data(&linked, root, ImageInterpolation::Linear)
            .unwrap_err()
            .code,
        ErrorCode::FileNotFound
    );
    Ok(())
}
