//! Native material texture fields decode their image through the image's colourspace, as Blender's Image
//! Texture does: an sRGB image is linearised, and a Non-Color image is read as stored.

use std::{collections::BTreeMap, error::Error};

use potter_core::{
    image::{ImageData, ImageInterpolation, load_image_data},
    model::{Id, SceneDoc},
    ops::apply_batch_with_project_context,
    shader::{HitContext, evaluate_surface},
};
use serde_json::json;

type TestResult<T = ()> = Result<T, Box<dyn Error>>;

const SCENE_ID: &str = "00000000-0000-4000-8000-000000000000";

/// The sRGB transfer function (IEC 61966-2-1). It is written out here, so the expectation does not reuse the
/// crate's decode.
fn srgb_to_linear(encoded: f64) -> f64 {
    if encoded <= 0.04045 {
        encoded / 12.92
    } else {
        ((encoded + 0.055) / 1.055).powf(2.4)
    }
}

#[test]
fn native_field_decodes_its_image_by_colourspace() -> TestResult {
    let root = tempfile::tempdir()?;
    // One mid-grey texel, stored as sRGB-encoded 128 in the file.
    let png = root.path().join("grey.png");
    ::image::RgbaImage::from_pixel(1, 1, ::image::Rgba([128, 128, 128, 255])).save(&png)?;
    let batch = json!({
        "schema_version": 1,
        "base_revision": 0,
        "operations": [
            {"op":"image.load","id":"data_img","path":png,"colorspace":"non_color"},
            {"op":"image.load","id":"colour_img","path":png,"colorspace":"srgb"},
            {"op":"material.create","id":"data_mat","roughness":1.0,"roughness_texture":{"image":"data_img"}},
            {"op":"material.create","id":"colour_mat","roughness":1.0,"roughness_texture":{"image":"colour_img"}}
        ]
    });
    let empty = SceneDoc::new(SCENE_ID.to_owned());
    let outcome = apply_batch_with_project_context(&empty, &batch, root.path(), None)?;
    for (digest, bytes) in &outcome.asset_blobs {
        let hex = digest
            .strip_prefix("sha256:")
            .ok_or("image blob digest is malformed")?;
        let blob = root
            .path()
            .join("assets")
            .join("sha256")
            .join(hex)
            .join("blob");
        std::fs::create_dir_all(blob.parent().ok_or("image blob parent is missing")?)?;
        std::fs::write(blob, bytes)?;
    }
    let doc = outcome.doc;
    let images = doc
        .images
        .iter()
        .map(|(id, image)| -> TestResult<(String, ImageData)> {
            Ok((
                id.as_str().to_owned(),
                load_image_data(image, root.path(), ImageInterpolation::Linear)?,
            ))
        })
        .collect::<TestResult<BTreeMap<String, ImageData>>>()?;
    let roughness = |material: &str| -> TestResult<f64> {
        let material = doc
            .materials
            .get(&Id::new(material)?)
            .ok_or("material is missing")?;
        let context = HitContext::new(&doc.node_groups)
            .with_images(&images)
            .with_path_traced(true);
        let shading = evaluate_surface(material, &context)?;
        Ok(shading
            .dominant()
            .ok_or("material has no closure")?
            .roughness)
    };
    let encoded = 128.0 / 255.0;
    let data = roughness("data_mat")?;
    let colour = roughness("colour_mat")?;
    assert!(
        (data - encoded).abs() < 1e-6,
        "a Non-Color image is read as stored: expected {encoded}, got {data}"
    );
    assert!(
        (colour - srgb_to_linear(encoded)).abs() < 1e-6,
        "an sRGB image is linearised: expected {}, got {colour}",
        srgb_to_linear(encoded)
    );
    Ok(())
}
