#![expect(
    clippy::float_cmp,
    reason = "asserting deterministic compositor pixels"
)]
#![expect(clippy::unwrap_used, reason = "small compositor fixtures")]
#![expect(
    clippy::needless_pass_by_value,
    reason = "graph fixtures clone JSON properties into isolated nodes"
)]

use std::{error::Error, fs, io::Cursor};

use serde_json::{Value, json};
use tempfile::tempdir;

use super::{Image, evaluate_graph};
use crate::{
    error::{ErrorCode, Result},
    graph::{GraphKind, GraphLink, GraphNode, NodeGroup},
    model::{Id, SceneDoc},
    render::raster::RasterOutput,
};

const BASE_PIXEL: [f32; 4] = [0.2, 0.4, 0.6, 0.5];

fn test_node(node_type: &str, properties: Value) -> GraphNode {
    let mut node = GraphNode::new(node_type);
    node.properties = properties.as_object().unwrap().clone();
    node
}

fn graph_with_node(node: GraphNode, output_socket: &str) -> NodeGroup {
    let mut graph = NodeGroup::new("compositor test", GraphKind::Compositor);
    let source_id = Id::from_static("source");
    let output_id = Id::from_static("z_output");
    graph.nodes.insert(source_id.clone(), node);
    graph
        .nodes
        .insert(output_id.clone(), GraphNode::new("CompositorNodeComposite"));
    graph.links.push(GraphLink {
        from_node: source_id,
        from_socket: output_socket.to_owned(),
        to_node: output_id,
        to_socket: "Image".to_owned(),
    });
    graph
}

fn run_node(
    node: GraphNode,
    output_socket: &str,
    base: &Image,
    passes: Option<&RasterOutput>,
    doc: &SceneDoc,
) -> Result<Image> {
    evaluate_graph(
        &graph_with_node(node, output_socket),
        base,
        passes,
        doc,
        1.0,
    )
}

fn assert_pixel(actual: [f32; 4], expected: [f32; 4]) {
    for (actual, expected) in actual.into_iter().zip(expected) {
        assert!(
            (actual - expected).abs() < 1.0e-5,
            "expected {expected}, got {actual}"
        );
    }
}

fn assert_node_types(
    doc: &SceneDoc,
    base: &Image,
    node_types: &[&str],
    properties: Value,
    expected: [f32; 4],
) -> Result<()> {
    for node_type in node_types {
        let output = run_node(
            test_node(node_type, properties.clone()),
            "Image",
            base,
            None,
            doc,
        )?;
        assert_eq!((output.width, output.height), (1, 1), "{node_type}");
        assert_pixel(output.pixels[0], expected);
    }
    Ok(())
}

fn raster_passes() -> RasterOutput {
    RasterOutput {
        rgba: vec![0; 4],
        linear_rgba: vec![BASE_PIXEL],
        ids: vec![7],
        depths: vec![4.0],
        elements: vec![11],
        normals: vec![[0.25, 0.5, 0.75]],
        albedo: vec![[0.1, 0.2, 0.3, 0.4]],
        emission: vec![[0.5, 0.4, 0.3, 0.2]],
        ambient_occlusion: vec![0.75],
    }
}

#[test]
fn supported_pixel_nodes_produce_their_numeric_results() -> Result<()> {
    let doc = SceneDoc::default();
    let base = Image::new(1, 1, vec![BASE_PIXEL])?;

    assert_node_types(
        &doc,
        &base,
        &[
            "CompositorNodeComposite",
            "Composite",
            "CompositorNodeViewer",
            "Viewer",
        ],
        json!({}),
        BASE_PIXEL,
    )?;
    assert_node_types(
        &doc,
        &base,
        &["CompositorNodeMixRGB", "MixRGB"],
        json!({
            "blend_type": "add",
            "color1": [0.2, 0.4, 0.6],
            "color2": [0.1, 0.1, 0.1],
            "factor": 0.5
        }),
        [0.25, 0.45, 0.65, 1.5],
    )?;
    assert_node_types(
        &doc,
        &base,
        &["CompositorNodeAlphaOver", "AlphaOver"],
        json!({
            "foreground": [1.0, 0.0, 0.0],
            "background": [0.0, 0.0, 1.0],
            "factor": 0.25
        }),
        [0.25, 0.0, 0.75, 1.0],
    )?;
    assert_node_types(
        &doc,
        &base,
        &[
            "CompositorNodeBlur",
            "Blur",
            "CompositorNodeDefocus",
            "Defocus",
        ],
        json!({"radius": 1}),
        BASE_PIXEL,
    )?;
    assert_node_types(
        &doc,
        &base,
        &["CompositorNodeGlare", "Glare"],
        json!({"threshold": 0.3, "radius": 0}),
        [0.2, 0.5, 0.9, 0.5],
    )?;
    assert_node_types(
        &doc,
        &base,
        &["CompositorNodeDenoise", "Denoise"],
        json!({}),
        BASE_PIXEL,
    )?;
    assert_node_types(
        &doc,
        &base,
        &["CompositorNodeExposure", "Exposure"],
        json!({"exposure": 1.0}),
        [0.4, 0.8, 1.2, 0.5],
    )?;
    assert_node_types(
        &doc,
        &base,
        &["CompositorNodeGamma", "Gamma"],
        json!({"gamma": 2.0}),
        [0.447_213_6, 0.632_455_5, 0.774_596_7, 0.5],
    )?;
    assert_node_types(
        &doc,
        &base,
        &["CompositorNodeInvert", "Invert"],
        json!({"factor": 0.25}),
        [0.35, 0.45, 0.55, 0.5],
    )?;
    assert_node_types(
        &doc,
        &base,
        &[
            "CompositorNodeBrightContrast",
            "BrightContrast",
            "CompositorNodeBrightnessContrast",
            "BrightnessContrast",
        ],
        json!({"brightness": 0.1, "contrast": 0.5}),
        [0.15, 0.45, 0.75, 0.5],
    )?;
    assert_node_types(
        &doc,
        &base,
        &["CompositorNodeColorBalance", "ColorBalance"],
        json!({"lift": [1.0, 1.0, 1.0], "gamma": [1.0, 1.0, 1.0], "gain": [2.0, 0.5, 1.5]}),
        [0.4, 0.2, 0.9, 0.5],
    )?;
    assert_node_types(
        &doc,
        &base,
        &["CompositorNodeCurves", "Curves"],
        json!({"curve": [[0.0, 0.0], [1.0, 0.5]]}),
        [0.1, 0.2, 0.3, 0.5],
    )?;
    assert_node_types(
        &doc,
        &base,
        &["CompositorNodeHueSat", "HueSat"],
        json!({"value": 0.5}),
        [0.1, 0.2, 0.3, 0.5],
    )?;
    assert_node_types(
        &doc,
        &base,
        &["CompositorNodeMath", "Math"],
        json!({"operation": "subtract", "value1": 5.0, "value2": 2.0}),
        [3.0, 3.0, 3.0, 1.0],
    )?;
    assert_node_types(
        &doc,
        &base,
        &["CompositorNodeSetAlpha", "SetAlpha"],
        json!({"alpha": 0.25}),
        [0.2, 0.4, 0.6, 0.25],
    )?;
    assert_node_types(
        &doc,
        &base,
        &["CompositorNodeMask", "Mask"],
        json!({"mask": [0.25]}),
        [0.25, 0.25, 0.25, 1.0],
    )?;

    let passes = raster_passes();
    for node_type in ["CompositorNodeIDMask", "IDMask"] {
        let id_mask = run_node(
            test_node(node_type, json!({"index": 7})),
            "Image",
            &base,
            Some(&passes),
            &doc,
        )?;
        assert_pixel(id_mask.pixels[0], [1.0; 4]);
    }
    for node_type in ["CompositorNodeCryptomatte", "Cryptomatte"] {
        let cryptomatte_mask = run_node(
            test_node(node_type, json!({"index": 7})),
            "Image",
            &base,
            Some(&passes),
            &doc,
        )?;
        assert_pixel(cryptomatte_mask.pixels[0], [1.0; 4]);
    }

    let unsupported = run_node(
        GraphNode::new("CompositorNodeUnknown"),
        "Image",
        &base,
        None,
        &doc,
    )
    .unwrap_err();
    assert_eq!(unsupported.code, ErrorCode::UnsupportedFeature);
    Ok(())
}

#[test]
fn render_layer_node_exposes_numeric_pass_pixels() -> Result<()> {
    let doc = SceneDoc::default();
    let base = Image::new(1, 1, vec![BASE_PIXEL])?;
    let passes = raster_passes();
    for node_type in ["CompositorNodeRLayers", "RLayers"] {
        for (socket, expected) in [
            ("Image", BASE_PIXEL),
            ("Z", [4.0, 4.0, 4.0, 1.0]),
            ("IndexOB", [7.0, 7.0, 7.0, 1.0]),
            ("Normal", [0.25, 0.5, 0.75, 1.0]),
            ("DiffCol", [0.1, 0.2, 0.3, 0.4]),
            ("Emit", [0.5, 0.4, 0.3, 0.2]),
            ("AO", [0.75, 0.75, 0.75, 1.0]),
        ] {
            let output = run_node(
                GraphNode::new(node_type),
                socket,
                &base,
                Some(&passes),
                &doc,
            )?;
            assert_pixel(output.pixels[0], expected);
        }
        let hashed = run_node(
            GraphNode::new(node_type),
            "Cryptomatte",
            &base,
            Some(&passes),
            &doc,
        )?;
        assert_pixel(
            hashed.pixels[0],
            [0.580_255_6, 0.580_255_6, 0.580_255_6, 1.0],
        );
    }
    Ok(())
}

#[test]
fn spatial_nodes_keep_dimensions_and_place_pixels_correctly() -> Result<()> {
    let doc = SceneDoc::default();
    let source = Image::new(
        2,
        2,
        vec![
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
        ],
    )?;
    for node_type in ["CompositorNodeScale", "Scale"] {
        let scaled = run_node(
            test_node(node_type, json!({"width": 4, "height": 2})),
            "Image",
            &source,
            None,
            &doc,
        )?;
        assert_eq!((scaled.width, scaled.height), (4, 2));
        assert_eq!(
            scaled.pixels,
            vec![
                source.pixels[0],
                source.pixels[0],
                source.pixels[1],
                source.pixels[1],
                source.pixels[2],
                source.pixels[2],
                source.pixels[3],
                source.pixels[3],
            ]
        );
    }

    for node_type in ["CompositorNodeTranslate", "Translate"] {
        let translated = run_node(
            test_node(node_type, json!({"x": 1, "y": 0})),
            "Image",
            &source,
            None,
            &doc,
        )?;
        assert_eq!(translated.pixels[0], [0.0; 4]);
        assert_eq!(translated.pixels[1], source.pixels[0]);
        assert_eq!(translated.pixels[2], [0.0; 4]);
        assert_eq!(translated.pixels[3], source.pixels[2]);
    }

    for node_type in ["CompositorNodeCrop", "Crop"] {
        let cropped = run_node(
            test_node(node_type, json!({"x": 1, "y": 0, "width": 1, "height": 2})),
            "Image",
            &source,
            None,
            &doc,
        )?;
        assert_eq!((cropped.width, cropped.height), (1, 2));
        assert_eq!(cropped.pixels, vec![source.pixels[1], source.pixels[3]]);
    }
    Ok(())
}

#[test]
fn output_file_node_writes_a_numeric_png_and_preserves_its_image() -> Result<(), Box<dyn Error>> {
    let directory = tempdir()?;
    let path = directory.path().join("composite.png");
    let base = Image::new(1, 1, vec![BASE_PIXEL])?;
    for node_type in ["CompositorNodeOutputFile", "OutputFile"] {
        let output = run_node(
            test_node(node_type, json!({"path": path.to_string_lossy()})),
            "Image",
            &base,
            None,
            &SceneDoc::default(),
        )?;
        assert_eq!(output.pixels, vec![BASE_PIXEL]);
    }
    let decoder = png::Decoder::new(Cursor::new(fs::read(path)?));
    let mut reader = decoder.read_info()?;
    let mut bytes = vec![0; reader.output_buffer_size()];
    reader.next_frame(&mut bytes)?;
    assert_eq!(&bytes[..4], &[51, 102, 153, 128]);
    Ok(())
}

#[test]
fn blend_modes_produce_numeric_rgba_samples() {
    let first = [0.25, 0.75, 0.5, 0.4];
    let second = [0.5, 0.25, 0.75, 0.2];
    assert_pixel(super::mix_rgb(super::BlendMode::Mix, first, second), second);
    assert_pixel(
        super::mix_rgb(super::BlendMode::Add, first, second),
        [0.75, 1.0, 1.25, 0.6],
    );
    assert_pixel(
        super::mix_rgb(super::BlendMode::Multiply, first, second),
        [0.125, 0.1875, 0.375, 0.08],
    );
    assert_pixel(
        super::mix_rgb(super::BlendMode::Screen, first, second),
        [0.625, 0.8125, 0.875, 0.52],
    );
    assert_pixel(
        super::mix_rgb(super::BlendMode::Overlay, first, second),
        [0.25, 0.625, 0.75, 0.16],
    );
}
#[test]
fn image_constructors_reject_empty_dimensions_and_mismatched_pixels() {
    assert_eq!(
        Image::new(0, 1, Vec::new()).unwrap_err().code,
        ErrorCode::InvalidArgument
    );
    assert_eq!(
        Image::new(1, 1, Vec::new()).unwrap_err().code,
        ErrorCode::InvalidArgument
    );
    assert_eq!(
        Image::filled(1, 0, BASE_PIXEL).unwrap_err().code,
        ErrorCode::InvalidArgument
    );
}

#[test]
fn gaussian_blur_zero_radius_is_identity_and_positive_radius_spreads_an_impulse() -> Result<()> {
    let image = Image::new(
        5,
        1,
        (0..5)
            .map(|index| {
                let value = if index == 2 { 1.0 } else { 0.0 };
                [value, value, value, 1.0]
            })
            .collect(),
    )?;
    assert_eq!(super::gaussian_blur(&image, 0)?, image);
    let blurred = super::gaussian_blur(&image, 1)?;
    assert!(blurred.pixels[1][0] > 0.0);
    assert!(blurred.pixels[2][0] < 1.0);
    assert!(blurred.pixels[3][0] > 0.0);
    Ok(())
}
