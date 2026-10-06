mod image;
#[cfg(test)]
mod tests;

use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::{Path, PathBuf},
};

use serde_json::Value;

use crate::{
    color::{ColorManagement, linear_to_srgb, sample_curve},
    error::{ErrorCode, PotError, Result},
    graph::{GraphKind, GraphNode, NodeGroup},
    model::{Id, Scene, SceneDoc},
    render::raster::RasterOutput,
};

pub use image::{BlendMode, Image, gaussian_blur, mix_rgb};

/// Applies compositor nodes (when enabled) and the selected display transform to render output.
pub(crate) fn apply(
    output: &mut RasterOutput,
    doc: &SceneDoc,
    scene: &Scene,
    width: u32,
    height: u32,
    frame: f64,
) -> Result<()> {
    validate_render_output(output, width, height)?;
    if scene.use_compositing
        && let Some(graph_id) = &scene.compositor
    {
        let graph = doc.node_groups.get(graph_id).ok_or_else(|| {
            PotError::with_details(
                ErrorCode::TargetNotFound,
                "scene compositor graph was not found",
                serde_json::json!({"graph": graph_id}),
            )
        })?;
        if graph.kind != GraphKind::Compositor {
            return Err(PotError::new(
                ErrorCode::InvalidOperation,
                "scene compositor reference must identify a compositor node group",
            ));
        }
        let base = Image {
            width,
            height,
            pixels: output.linear_rgba.clone(),
        };
        let composite = evaluate_graph(graph, &base, Some(output), doc, frame)?;
        output.linear_rgba = composite.pixels;
    }
    encode_display(
        &mut output.rgba,
        &output.linear_rgba,
        &scene.color_management,
    )?;
    Ok(())
}

/// Applies the selected scene compositor to an already-linear image without display encoding.
pub(crate) fn apply_image(
    image: Image,
    doc: &SceneDoc,
    scene: &Scene,
    frame: f64,
) -> Result<Image> {
    if !scene.use_compositing {
        return Ok(image);
    }
    let Some(graph_id) = &scene.compositor else {
        return Ok(image);
    };
    let graph = doc.node_groups.get(graph_id).ok_or_else(|| {
        PotError::with_details(
            ErrorCode::TargetNotFound,
            "scene compositor graph was not found",
            serde_json::json!({"graph": graph_id}),
        )
    })?;
    if graph.kind != GraphKind::Compositor {
        return Err(PotError::new(
            ErrorCode::InvalidOperation,
            "scene compositor reference must identify a compositor node group",
        ));
    }
    evaluate_graph(graph, &image, None, doc, frame)
}

enum NodeOutput {
    Image(Image),
    RenderLayers,
}

fn evaluate_graph(
    graph: &NodeGroup,
    base: &Image,
    passes: Option<&RasterOutput>,
    doc: &SceneDoc,
    frame: f64,
) -> Result<Image> {
    let mut pending: BTreeSet<Id> = graph.nodes.keys().cloned().collect();
    let mut rendered: BTreeMap<Id, NodeOutput> = BTreeMap::new();
    let mut composite = None;
    let mut viewer = None;

    while !pending.is_empty() {
        let mut completed = Vec::new();
        for node_id in &pending {
            let node = graph.nodes.get(node_id).ok_or_else(|| {
                PotError::new(ErrorCode::InternalError, "compositor node disappeared")
            })?;
            let mut inputs = BTreeMap::new();
            let mut ready = true;
            for link in graph.links.iter().filter(|link| &link.to_node == node_id) {
                let Some(source) = rendered.get(&link.from_node) else {
                    ready = false;
                    break;
                };
                let image = match source {
                    NodeOutput::Image(image) => image.clone(),
                    NodeOutput::RenderLayers => {
                        let source_node = graph.nodes.get(&link.from_node).ok_or_else(|| {
                            PotError::new(
                                ErrorCode::InternalError,
                                "compositor render-layer node disappeared",
                            )
                        })?;
                        render_layer(source_node, &link.from_socket, base, passes)?
                    }
                };
                inputs.insert(link.to_socket.clone(), image);
            }
            if !ready {
                continue;
            }
            if is_type(node, "CompositorNodeRLayers") || is_type(node, "RLayers") {
                rendered.insert(node_id.clone(), NodeOutput::RenderLayers);
                completed.push(node_id.clone());
                continue;
            }
            let image = input_image(node, &inputs, base)?;
            let output = match node.node_type.as_str() {
                "CompositorNodeComposite" | "Composite" => {
                    composite = Some(image.clone());
                    image
                }
                "CompositorNodeViewer" | "Viewer" => {
                    viewer = Some(image.clone());
                    image
                }
                "CompositorNodeOutputFile" | "OutputFile" => {
                    write_file_output(node, &image)?;
                    image
                }
                "CompositorNodeMixRGB" | "MixRGB" => mix_images(node, &inputs, base)?,
                "CompositorNodeAlphaOver" | "AlphaOver" => alpha_over(node, &inputs, base)?,
                "CompositorNodeBlur" | "Blur" => {
                    let radius = property_u32(node, &["size_x", "radius"], 1)?.min(256);
                    gaussian_blur(&image, radius)?
                }
                "CompositorNodeGlare" | "Glare" => glare(node, &image)?,
                "CompositorNodeDenoise" | "Denoise" => {
                    // This edge-aware bilateral filter is a deterministic denoiser, not a
                    // Monte Carlo variance estimator or replacement for a production denoiser.
                    bilateral_denoise(&image)?
                }
                "CompositorNodeDefocus" | "Defocus" => {
                    let radius = property_u32(node, &["radius", "size"], 2)?.min(256);
                    gaussian_blur(&image, radius)?
                }
                "CompositorNodeExposure" | "Exposure" => {
                    let exposure =
                        scalar_default_value(node, &["Exposure"], &["exposure", "value"], 0.0)?;
                    map_rgb(&image, |value| value * 2.0_f32.powf(exposure))
                }
                "CompositorNodeGamma" | "Gamma" => {
                    let gamma = scalar_default_value(node, &["Gamma"], &["gamma", "value"], 1.0)?;
                    if gamma <= 0.0 {
                        return Err(PotError::invalid_argument(
                            "compositor gamma must be positive and finite",
                        ));
                    }
                    map_rgb(&image, |value| value.max(0.0).powf(1.0 / gamma))
                }
                "CompositorNodeInvert" | "Invert" => {
                    let factor = scalar_default_value(node, &["Fac"], &["factor", "fac"], 1.0)?
                        .clamp(0.0, 1.0);
                    map_rgb(&image, |value| {
                        value * (1.0 - factor) + (1.0 - value) * factor
                    })
                }
                "CompositorNodeBrightContrast"
                | "CompositorNodeBrightnessContrast"
                | "BrightContrast"
                | "BrightnessContrast" => {
                    let brightness = scalar_default_value(
                        node,
                        &["Bright", "Brightness"],
                        &["brightness"],
                        0.0,
                    )?;
                    let contrast = scalar_default_value(node, &["Contrast"], &["contrast"], 0.0)?;
                    map_rgb(&image, |value| {
                        (value - 0.5) * (1.0 + contrast) + 0.5 + brightness
                    })
                }
                "CompositorNodeColorBalance" | "ColorBalance" => {
                    let lift = color_property(node, "lift", [1.0; 3])?;
                    let gamma = color_property(node, "gamma", [1.0; 3])?;
                    let gain = color_property(node, "gain", [1.0; 3])?;
                    if gamma.iter().any(|channel| *channel <= 0.0) {
                        return Err(PotError::invalid_argument(
                            "ColorBalance gamma channels must be positive",
                        ));
                    }
                    map_color_balance(&image, lift, gamma, gain)
                }
                "CompositorNodeCurves" | "Curves" => {
                    let curve = curve_property(node)?;
                    map_rgb(&image, |value| {
                        if curve.is_empty() {
                            value
                        } else {
                            sample_curve(f64::from(value), &curve) as f32
                        }
                    })
                }
                "CompositorNodeHueSat" | "HueSat" => {
                    let hue = scalar_default_value(node, &["Hue"], &["hue"], 0.5)?;
                    let saturation =
                        scalar_default_value(node, &["Saturation"], &["saturation"], 1.0)?;
                    let value = scalar_default_value(node, &["Value"], &["value"], 1.0)?;
                    map_hue_sat(&image, hue, saturation, value)
                }
                "CompositorNodeMath" | "Math" => math_node(node, &inputs, base)?,
                "CompositorNodeScale" | "Scale" => {
                    let width = property_u32(node, &["width"], image.width)?.max(1);
                    let height = property_u32(node, &["height"], image.height)?.max(1);
                    scale_nearest(&image, width, height)?
                }
                "CompositorNodeTranslate" | "Translate" => {
                    let x = property_i32(node, &["x", "x_offset"], 0)?;
                    let y = property_i32(node, &["y", "y_offset"], 0)?;
                    translate(&image, x, y)?
                }
                "CompositorNodeCrop" | "Crop" => crop(node, &image)?,
                "CompositorNodeSetAlpha" | "SetAlpha" => set_alpha(node, &image, &inputs)?,
                "CompositorNodeIDMask" | "IDMask" | "CompositorNodeCryptomatte" | "Cryptomatte" => {
                    id_mask(node, passes, base)?
                }
                "CompositorNodeMask" | "Mask" => mask_input(node, base, doc, frame)?,
                _ => {
                    return Err(PotError::with_details(
                        ErrorCode::UnsupportedFeature,
                        format!("compositor node `{}` is not supported", node.node_type),
                        serde_json::json!({"feature_id": format!("compositor.node.{}", node.node_type)}),
                    ));
                }
            };
            rendered.insert(node_id.clone(), NodeOutput::Image(output));
            completed.push(node_id.clone());
        }
        if completed.is_empty() {
            return Err(PotError::new(
                ErrorCode::InvalidOperation,
                "compositor graph contains a cycle or an unresolved link",
            ));
        }
        for node_id in completed {
            pending.remove(&node_id);
        }
    }
    Ok(composite.or(viewer).unwrap_or_else(|| base.clone()))
}

fn render_layer(
    node: &GraphNode,
    socket: &str,
    base: &Image,
    passes: Option<&RasterOutput>,
) -> Result<Image> {
    let requested = node
        .inputs
        .get("pass")
        .and_then(Value::as_str)
        .or_else(|| node.properties.get("pass").and_then(Value::as_str))
        .unwrap_or(socket);
    match requested {
        "" | "Image" | "image" | "Combined" | "combined" => Ok(base.clone()),
        "Z" | "z" | "Depth" | "depth" => {
            let passes = passes.ok_or_else(|| unsupported_pass("z"))?;
            Image::new(
                base.width,
                base.height,
                passes
                    .depths
                    .iter()
                    .map(|depth| [*depth, *depth, *depth, 1.0])
                    .collect(),
            )
        }
        "IndexOB" | "object_index" | "ObjectIndex" => {
            let passes = passes.ok_or_else(|| unsupported_pass("object_index"))?;
            Image::new(
                base.width,
                base.height,
                passes
                    .ids
                    .iter()
                    .map(|id| {
                        let value = *id as f32;
                        [value, value, value, if *id == 0 { 0.0 } else { 1.0 }]
                    })
                    .collect(),
            )
        }
        "Cryptomatte" | "CryptoObject" | "cryptomatte" => {
            let passes = passes.ok_or_else(|| unsupported_pass("cryptomatte"))?;
            Image::new(
                base.width,
                base.height,
                passes
                    .ids
                    .iter()
                    .map(|id| {
                        let value = (cryptomatte_hash(*id) as f32) / (u32::MAX as f32);
                        [value, value, value, if *id == 0 { 0.0 } else { 1.0 }]
                    })
                    .collect(),
            )
        }
        "Normal" | "normal" => {
            let passes = passes.ok_or_else(|| unsupported_pass("normal"))?;
            Image::new(
                base.width,
                base.height,
                passes
                    .normals
                    .iter()
                    .map(|normal| [normal[0], normal[1], normal[2], 1.0])
                    .collect(),
            )
        }
        "DiffCol" | "Albedo" | "albedo" => {
            let passes = passes.ok_or_else(|| unsupported_pass("albedo"))?;
            Image::new(base.width, base.height, passes.albedo.clone())
        }
        "Emit" | "Emission" | "emission" => {
            let passes = passes.ok_or_else(|| unsupported_pass("emission"))?;
            Image::new(base.width, base.height, passes.emission.clone())
        }
        "ao" | "AO" => {
            let passes = passes.ok_or_else(|| unsupported_pass("ao"))?;
            Image::new(
                base.width,
                base.height,
                passes
                    .ambient_occlusion
                    .iter()
                    .map(|value| [*value, *value, *value, 1.0])
                    .collect(),
            )
        }
        pass => Err(unsupported_pass(pass)),
    }
}

fn unsupported_pass(pass: &str) -> PotError {
    PotError::with_details(
        ErrorCode::UnsupportedFeature,
        format!("render pass `{pass}` is unavailable in this compositor context"),
        serde_json::json!({"feature_id": format!("render_pass.{pass}")}),
    )
}

fn input_image(node: &GraphNode, inputs: &BTreeMap<String, Image>, base: &Image) -> Result<Image> {
    for key in [
        "Image",
        "image",
        "Color",
        "color",
        "Value",
        "Value_001",
        "Mask",
    ] {
        if let Some(image) = inputs.get(key) {
            return Ok(image.clone());
        }
    }
    if let Some(image) = inputs.values().next() {
        return Ok(image.clone());
    }
    for key in ["Image", "image", "Color", "color"] {
        if node.inputs.contains_key(key) {
            let color = color_property(node, key, [0.0; 3])?;
            return Image::filled(base.width, base.height, [color[0], color[1], color[2], 1.0]);
        }
    }
    Ok(base.clone())
}

fn mix_images(node: &GraphNode, inputs: &BTreeMap<String, Image>, base: &Image) -> Result<Image> {
    let first = match inputs.get("Image1").or_else(|| inputs.get("A")) {
        Some(image) => image.clone(),
        None => color_image(node, "color1", base, [0.5; 3])?,
    };
    let second = match inputs.get("Image2").or_else(|| inputs.get("B")) {
        Some(image) => image.clone(),
        None => color_image(node, "color2", base, [0.5; 3])?,
    };
    ensure_same_dimensions(&first, &second)?;
    let blend = property_text(node, &["blend_type", "mode"])?.unwrap_or("mix");
    let mode = match blend {
        "mix" | "MIX" => BlendMode::Mix,
        "add" | "ADD" => BlendMode::Add,
        "multiply" | "MULTIPLY" => BlendMode::Multiply,
        "screen" | "SCREEN" => BlendMode::Screen,
        "overlay" | "OVERLAY" => BlendMode::Overlay,
        _ => {
            return Err(PotError::with_details(
                ErrorCode::UnsupportedFeature,
                format!("MixRGB blend mode `{blend}` is unsupported"),
                serde_json::json!({"feature_id": format!("compositor.mix_rgb.{blend}")}),
            ));
        }
    };
    let factor =
        scalar_default_value(node, &["Fac", "Factor"], &["factor", "fac"], 1.0)?.clamp(0.0, 1.0);
    let pixels = first
        .pixels
        .iter()
        .zip(&second.pixels)
        .map(|(a, b)| {
            let mixed = mix_rgb(mode, *a, *b);
            std::array::from_fn(|channel| a[channel] * (1.0 - factor) + mixed[channel] * factor)
        })
        .collect();
    Image::new(first.width, first.height, pixels)
}

fn alpha_over(node: &GraphNode, inputs: &BTreeMap<String, Image>, base: &Image) -> Result<Image> {
    let foreground = match inputs.get("Image").or_else(|| inputs.get("Foreground")) {
        Some(image) => image.clone(),
        None => color_image(node, "foreground", base, [0.0; 3])?,
    };
    let background = match inputs.get("Image_001").or_else(|| inputs.get("Background")) {
        Some(image) => image.clone(),
        None => color_image(node, "background", base, [0.0; 3])?,
    };
    ensure_same_dimensions(&foreground, &background)?;
    let factor =
        scalar_default_value(node, &["Fac", "factor"], &["factor", "fac"], 1.0)?.clamp(0.0, 1.0);
    let pixels = foreground
        .pixels
        .iter()
        .zip(&background.pixels)
        .map(|(front, back)| {
            let alpha = front[3].clamp(0.0, 1.0) * factor;
            [
                front[0] * alpha + back[0] * (1.0 - alpha),
                front[1] * alpha + back[1] * (1.0 - alpha),
                front[2] * alpha + back[2] * (1.0 - alpha),
                alpha + back[3] * (1.0 - alpha),
            ]
        })
        .collect();
    Image::new(foreground.width, foreground.height, pixels)
}

fn glare(node: &GraphNode, image: &Image) -> Result<Image> {
    let glare_type = property_text(node, &["type", "glare_type"])?.unwrap_or("FOG_GLOW");
    if !matches!(glare_type, "fog_glow" | "FOG_GLOW" | "glow" | "GLOW") {
        return Err(PotError::with_details(
            ErrorCode::UnsupportedFeature,
            format!("Glare type `{glare_type}` is unsupported"),
            serde_json::json!({"feature_id": format!("compositor.glare.{glare_type}")}),
        ));
    }
    let threshold = scalar_default_value(node, &["Threshold"], &["threshold"], 1.0)?;
    let radius = property_u32(node, &["size", "radius"], 4)?.min(64);
    let bright = map_rgb(image, |channel| (channel - threshold).max(0.0));
    let bloom = gaussian_blur(&bright, radius)?;
    Ok(Image {
        width: image.width,
        height: image.height,
        pixels: image
            .pixels
            .iter()
            .zip(bloom.pixels)
            .map(|(base, glow)| {
                std::array::from_fn(|channel| {
                    if channel == 3 {
                        base[channel]
                    } else {
                        base[channel] + glow[channel]
                    }
                })
            })
            .collect(),
    })
}

fn bilateral_denoise(image: &Image) -> Result<Image> {
    let radius = 1_i64;
    let width = i64::from(image.width);
    let height = i64::from(image.height);
    let mut pixels = Vec::with_capacity(image.pixels.len());
    for y in 0..height {
        for x in 0..width {
            let center = image.pixels[(y * width + x) as usize];
            let mut accumulated = [0.0_f32; 4];
            let mut total = 0.0_f32;
            for dy in -radius..=radius {
                for dx in -radius..=radius {
                    let sample_x = (x + dx).rem_euclid(width);
                    let sample_y = (y + dy).rem_euclid(height);
                    let sample = image.pixels[(sample_y * width + sample_x) as usize];
                    let distance = (0..3)
                        .map(|channel| (center[channel] - sample[channel]).powi(2))
                        .sum::<f32>();
                    let weight = (-distance * 16.0).exp();
                    for channel in 0..4 {
                        accumulated[channel] += sample[channel] * weight;
                    }
                    total += weight;
                }
            }
            pixels.push(accumulated.map(|channel| channel / total));
        }
    }
    Image::new(image.width, image.height, pixels)
}

fn math_node(node: &GraphNode, inputs: &BTreeMap<String, Image>, base: &Image) -> Result<Image> {
    let first_default = scalar_default_image(node, &["Value"], &["value1", "value"], 0.0, base)?;
    let second_default =
        scalar_default_image(node, &["Value_001", "Value 2"], &["value2"], 0.0, base)?;
    let first = inputs
        .get("Value")
        .or_else(|| inputs.get("Value 1"))
        .or_else(|| inputs.get("A"))
        .unwrap_or(&first_default);
    let second = inputs
        .get("Value_001")
        .or_else(|| inputs.get("Value 2"))
        .or_else(|| inputs.get("B"))
        .unwrap_or(&second_default);
    ensure_same_dimensions(first, second)?;

    let operation = property_text(node, &["operation", "mode"])?.unwrap_or("add");
    let calculate: fn(f32, f32) -> f32 = match operation {
        "add" | "ADD" => |a, b| a + b,
        "subtract" | "SUBTRACT" => |a, b| a - b,
        "multiply" | "MULTIPLY" => |a, b| a * b,
        "divide" | "DIVIDE" => |a, b| if b == 0.0 { 0.0 } else { a / b },
        "power" | "POWER" => |a, b| a.powf(b),
        "maximum" | "MAXIMUM" => f32::max,
        "minimum" | "MINIMUM" => f32::min,
        "less_than" | "LESS_THAN" => |a, b| if a < b { 1.0 } else { 0.0 },
        "greater_than" | "GREATER_THAN" => |a, b| if a > b { 1.0 } else { 0.0 },
        "compare" | "COMPARE" => |a, b| {
            if (a - b).abs() <= f32::EPSILON {
                1.0
            } else {
                0.0
            }
        },
        _ => {
            return Err(PotError::with_details(
                ErrorCode::UnsupportedFeature,
                format!("Math operation `{operation}` is unsupported"),
                serde_json::json!({"feature_id": format!("compositor.math.{operation}")}),
            ));
        }
    };
    let pixels = first
        .pixels
        .iter()
        .zip(&second.pixels)
        .map(|(a, b)| {
            let value = calculate(a[0], b[0]);
            [value, value, value, 1.0]
        })
        .collect();
    Image::new(first.width, first.height, pixels)
}

fn scalar_default_image(
    node: &GraphNode,
    sockets: &[&str],
    properties: &[&str],
    default: f64,
    base: &Image,
) -> Result<Image> {
    let value = scalar_default_value(node, sockets, properties, default)?;
    Image::filled(base.width, base.height, [value, value, value, 1.0])
}

fn scalar_default_value(
    node: &GraphNode,
    sockets: &[&str],
    properties: &[&str],
    default: f64,
) -> Result<f32> {
    let mut value = None;
    for socket in sockets {
        if let Some(input) = node.inputs.get(*socket) {
            value = Some(
                input
                    .as_f64()
                    .filter(|number| number.is_finite())
                    .ok_or_else(|| {
                        PotError::invalid_argument("compositor scalar input must be finite")
                    })?,
            );
            break;
        }
    }
    let value = match value {
        Some(value) => value,
        None => property_f64(node, properties, default)?,
    };
    let value = value as f32;
    if !value.is_finite() {
        return Err(PotError::invalid_argument(
            "compositor scalar input exceeds f32 range",
        ));
    }
    Ok(value)
}

fn id_mask(node: &GraphNode, passes: Option<&RasterOutput>, base: &Image) -> Result<Image> {
    let use_hash = node.node_type.contains("Cryptomatte");
    let target = property_u32(node, &["index", "id", "object_index"], 1)?;
    let passes = passes.ok_or_else(|| {
        unsupported_pass(if use_hash {
            "cryptomatte"
        } else {
            "object_index"
        })
    })?;
    let hashed_target = cryptomatte_hash(target);
    Image::new(
        base.width,
        base.height,
        passes
            .ids
            .iter()
            .map(|id| {
                let matches = if use_hash {
                    cryptomatte_hash(*id) == hashed_target
                } else {
                    *id == target
                };
                let value = if matches { 1.0 } else { 0.0 };
                [value, value, value, 1.0]
            })
            .collect(),
    )
}

/// Stable 32-bit integer hash used for this engine's Cryptomatte-lite identifiers.
#[must_use]
pub const fn cryptomatte_hash(object_index: u32) -> u32 {
    let mut value = object_index;
    value ^= value >> 16;
    value = value.wrapping_mul(0x7feb_352d);
    value ^= value >> 15;
    value = value.wrapping_mul(0x846c_a68b);
    value ^ (value >> 16)
}

fn mask_input(node: &GraphNode, base: &Image, doc: &SceneDoc, frame: f64) -> Result<Image> {
    let mask_id = ["mask_id", "mask"].iter().find_map(|key| {
        node.inputs
            .get(*key)
            .and_then(Value::as_str)
            .or_else(|| node.properties.get(*key).and_then(Value::as_str))
    });
    if let Some(mask_id) = mask_id {
        let id = Id::new(mask_id.to_owned())?;
        let mask = doc.masks.get(&id).ok_or_else(|| {
            PotError::with_details(
                ErrorCode::TargetNotFound,
                "compositor mask node references a missing mask",
                serde_json::json!({"mask": mask_id}),
            )
        })?;
        let mask_image = mask
            .rasterize(base.width, base.height, frame)
            .map_err(|error| {
                PotError::invalid_argument(format!("mask rasterization failed: {error}"))
            })?;
        if mask_image.pixels.len() != base.pixels.len() {
            return Err(PotError::new(
                ErrorCode::RenderFailed,
                "rasterized mask dimensions do not match compositor image",
            ));
        }
        return Image::new(
            base.width,
            base.height,
            mask_image
                .pixels
                .into_iter()
                .map(|coverage| [coverage, coverage, coverage, 1.0])
                .collect(),
        );
    }

    let Some(values) = node
        .inputs
        .get("mask")
        .and_then(Value::as_array)
        .or_else(|| node.properties.get("mask").and_then(Value::as_array))
    else {
        return Err(PotError::invalid_argument(
            "compositor mask node requires a mask_id or explicit mask values",
        ));
    };
    if values.len() != base.pixels.len() {
        return Err(PotError::invalid_argument(
            "mask input length must match compositor image dimensions",
        ));
    }
    let pixels = values
        .iter()
        .map(|value| {
            let scalar = value
                .as_f64()
                .filter(|value| value.is_finite())
                .ok_or_else(|| PotError::invalid_argument("mask values must be finite numbers"))?
                as f32;
            Ok([scalar, scalar, scalar, 1.0])
        })
        .collect::<Result<Vec<_>>>()?;
    Image::new(base.width, base.height, pixels)
}

fn scale_nearest(image: &Image, width: u32, height: u32) -> Result<Image> {
    let mut pixels = Vec::with_capacity(
        usize::try_from(width)
            .unwrap_or(0)
            .saturating_mul(usize::try_from(height).unwrap_or(0)),
    );
    for y in 0..height {
        for x in 0..width {
            let source_x = (u64::from(x) * u64::from(image.width) / u64::from(width)) as usize;
            let source_y = (u64::from(y) * u64::from(image.height) / u64::from(height)) as usize;
            pixels.push(image.pixels[source_y * image.width as usize + source_x]);
        }
    }
    Image::new(width, height, pixels)
}

fn translate(image: &Image, offset_x: i32, offset_y: i32) -> Result<Image> {
    let width = i64::from(image.width);
    let height = i64::from(image.height);
    let mut pixels = vec![[0.0, 0.0, 0.0, 0.0]; image.pixels.len()];
    for y in 0..height {
        for x in 0..width {
            let target_x = x + i64::from(offset_x);
            let target_y = y + i64::from(offset_y);
            if (0..width).contains(&target_x) && (0..height).contains(&target_y) {
                pixels[(target_y * width + target_x) as usize] =
                    image.pixels[(y * width + x) as usize];
            }
        }
    }
    Image::new(image.width, image.height, pixels)
}

fn crop(node: &GraphNode, image: &Image) -> Result<Image> {
    let x = property_u32(node, &["x"], 0)?.min(image.width);
    let y = property_u32(node, &["y"], 0)?.min(image.height);
    let width = property_u32(node, &["width"], image.width.saturating_sub(x))?
        .min(image.width.saturating_sub(x));
    let height = property_u32(node, &["height"], image.height.saturating_sub(y))?
        .min(image.height.saturating_sub(y));
    if width == 0 || height == 0 {
        return Err(PotError::invalid_argument(
            "crop dimensions must be non-zero",
        ));
    }
    let mut pixels = Vec::with_capacity(width as usize * height as usize);
    for row in y..(y + height) {
        let start = row as usize * image.width as usize + x as usize;
        let end = start + width as usize;
        pixels.extend_from_slice(&image.pixels[start..end]);
    }
    Image::new(width, height, pixels)
}

/// Writes selected render passes as named `OpenEXR` layers.
pub(crate) fn write_multilayer_exr(
    path: &Path,
    output: &RasterOutput,
    width: u32,
    height: u32,
    requested_passes: &[String],
) -> Result<()> {
    use exr::prelude::{
        AnyChannel, AnyChannels, Encoding, FlatSamples, Image as ExrImage, ImageAttributes, Layer,
        LayerAttributes, WritableImage,
    };

    let width = usize::try_from(width).map_err(|_| {
        PotError::new(
            ErrorCode::LimitExceeded,
            "EXR width exceeds platform limits",
        )
    })?;
    let height = usize::try_from(height).map_err(|_| {
        PotError::new(
            ErrorCode::LimitExceeded,
            "EXR height exceeds platform limits",
        )
    })?;
    let expected = width.checked_mul(height).ok_or_else(|| {
        PotError::new(
            ErrorCode::LimitExceeded,
            "EXR dimensions exceed platform limits",
        )
    })?;
    if requested_passes.is_empty() {
        return Err(PotError::invalid_argument(
            "at least one render pass is required for EXR output",
        ));
    }
    let mut seen = BTreeSet::new();
    let mut layers: Vec<Layer<AnyChannels<FlatSamples>>> =
        Vec::with_capacity(requested_passes.len());

    for pass in requested_passes {
        if !seen.insert(pass.as_str()) {
            return Err(PotError::invalid_argument(format!(
                "duplicate render pass `{pass}`"
            )));
        }
        let (name, channels) = match pass.as_str() {
            "combined" => {
                validate_pass_length("combined", &output.linear_rgba, expected)?;
                let channel = |name, index| {
                    AnyChannel::new(
                        name,
                        FlatSamples::F32(
                            output
                                .linear_rgba
                                .iter()
                                .map(|pixel| pixel[index])
                                .collect(),
                        ),
                    )
                };
                (
                    "Combined",
                    vec![
                        channel("R", 0),
                        channel("G", 1),
                        channel("B", 2),
                        channel("A", 3),
                    ],
                )
            }
            "z" => {
                validate_pass_length("z", &output.depths, expected)?;
                (
                    "Depth",
                    vec![AnyChannel::new(
                        "Z",
                        FlatSamples::F32(output.depths.clone()),
                    )],
                )
            }
            "normal" => {
                validate_pass_length("normal", &output.normals, expected)?;
                let channels = ["X", "Y", "Z"].map(|name| {
                    let index = match name {
                        "X" => 0,
                        "Y" => 1,
                        _ => 2,
                    };
                    AnyChannel::new(
                        name,
                        FlatSamples::F32(
                            output.normals.iter().map(|normal| normal[index]).collect(),
                        ),
                    )
                });
                ("Normal", channels.into())
            }
            "albedo" => {
                validate_pass_length("albedo", &output.albedo, expected)?;
                let channel = |name, index| {
                    AnyChannel::new(
                        name,
                        FlatSamples::F32(output.albedo.iter().map(|pixel| pixel[index]).collect()),
                    )
                };
                (
                    "Albedo",
                    vec![
                        channel("R", 0),
                        channel("G", 1),
                        channel("B", 2),
                        channel("A", 3),
                    ],
                )
            }
            "emission" => {
                validate_pass_length("emission", &output.emission, expected)?;
                let channel = |name, index| {
                    AnyChannel::new(
                        name,
                        FlatSamples::F32(
                            output.emission.iter().map(|pixel| pixel[index]).collect(),
                        ),
                    )
                };
                (
                    "Emission",
                    vec![
                        channel("R", 0),
                        channel("G", 1),
                        channel("B", 2),
                        channel("A", 3),
                    ],
                )
            }
            "object_index" => {
                validate_pass_length("object_index", &output.ids, expected)?;
                (
                    "ObjectIndex",
                    vec![AnyChannel::new(
                        "IndexOB",
                        FlatSamples::U32(output.ids.clone()),
                    )],
                )
            }
            "cryptomatte" => {
                validate_pass_length("cryptomatte", &output.ids, expected)?;
                (
                    "Cryptomatte",
                    vec![AnyChannel::new(
                        "Object",
                        FlatSamples::U32(
                            output.ids.iter().map(|id| cryptomatte_hash(*id)).collect(),
                        ),
                    )],
                )
            }
            "ao" => {
                validate_pass_length("ao", &output.ambient_occlusion, expected)?;
                (
                    "AO",
                    vec![AnyChannel::new(
                        "AO",
                        FlatSamples::F32(output.ambient_occlusion.clone()),
                    )],
                )
            }
            _ => {
                return Err(PotError::invalid_argument(format!(
                    "unsupported render pass `{pass}`"
                )));
            }
        };
        let channels = AnyChannels::sort(channels.into());
        layers.push(Layer::new(
            (width, height),
            LayerAttributes::named(name),
            Encoding::FAST_LOSSLESS,
            channels,
        ));
    }

    ExrImage::from_layers(ImageAttributes::with_size((width, height)), layers)
        .write()
        .to_file(path)
        .map_err(|error| {
            PotError::new(
                ErrorCode::RenderFailed,
                format!("multilayer EXR encoding failed: {error}"),
            )
        })
}

fn validate_pass_length<T>(name: &str, values: &[T], expected: usize) -> Result<()> {
    if values.len() != expected {
        return Err(PotError::new(
            ErrorCode::RenderFailed,
            format!("render pass `{name}` dimensions do not match EXR output"),
        ));
    }
    Ok(())
}

fn write_file_output(node: &GraphNode, image: &Image) -> Result<()> {
    let path = property_text(node, &["path", "base_path", "file_path"])?.ok_or_else(|| {
        PotError::invalid_argument("compositor output file node requires a path property")
    })?;
    let path = PathBuf::from(path);
    if path
        .extension()
        .is_some_and(|extension| extension.eq_ignore_ascii_case("exr"))
    {
        exr::prelude::write_rgba_file(
            &path,
            image.width as usize,
            image.height as usize,
            |x, y| {
                let pixel = image.pixels[y * image.width as usize + x];
                (pixel[0], pixel[1], pixel[2], pixel[3])
            },
        )
        .map_err(|error| {
            PotError::new(
                ErrorCode::RenderFailed,
                format!("EXR output encoding failed: {error}"),
            )
        })?;
        return Ok(());
    }
    let mut data = Vec::with_capacity(image.pixels.len() * 4);
    for pixel in &image.pixels {
        for channel in pixel {
            data.push((channel.clamp(0.0, 1.0) * 255.0).round() as u8);
        }
    }
    let file = fs::File::create(path).map_err(|error| PotError::io(&error))?;
    let mut encoder = png::Encoder::new(file, image.width, image.height);
    encoder.set_color(png::ColorType::Rgba);
    encoder.set_depth(png::BitDepth::Eight);
    let mut writer = encoder.write_header().map_err(|error| {
        PotError::new(
            ErrorCode::RenderFailed,
            format!("PNG output header failed: {error}"),
        )
    })?;
    writer.write_image_data(&data).map_err(|error| {
        PotError::new(
            ErrorCode::RenderFailed,
            format!("PNG output encoding failed: {error}"),
        )
    })
}

/// Applies scene color management and returns the image's display-encoded RGBA8 pixels.
pub(crate) fn encode_image_display(image: &Image, settings: &ColorManagement) -> Result<Vec<u8>> {
    let length = image.pixels.len().checked_mul(4).ok_or_else(|| {
        PotError::new(
            ErrorCode::LimitExceeded,
            "display image dimensions exceed platform limits",
        )
    })?;
    let mut bytes = vec![0; length];
    encode_display(&mut bytes, &image.pixels, settings)?;
    Ok(bytes)
}

fn encode_display(bytes: &mut [u8], pixels: &[[f32; 4]], settings: &ColorManagement) -> Result<()> {
    if bytes.len() != pixels.len().saturating_mul(4) {
        return Err(PotError::new(
            ErrorCode::RenderFailed,
            "display image buffer dimensions do not match",
        ));
    }
    let (encoded_pixels, remainder) = bytes.as_chunks_mut::<4>();
    if !remainder.is_empty() {
        return Err(PotError::new(
            ErrorCode::RenderFailed,
            "display image buffer dimensions do not match",
        ));
    }
    for (pixel, linear) in encoded_pixels.iter_mut().zip(pixels) {
        let transformed = settings.transform_linear([
            f64::from(linear[0]),
            f64::from(linear[1]),
            f64::from(linear[2]),
        ]);
        for channel in 0..3 {
            let display = if settings.display_device == "sRGB" {
                linear_to_srgb(transformed[channel])
            } else {
                transformed[channel]
            };
            pixel[channel] = (display.clamp(0.0, 1.0) * 255.0).round() as u8;
        }
        pixel[3] = (linear[3].clamp(0.0, 1.0) * 255.0).round() as u8;
    }
    Ok(())
}

fn validate_render_output(output: &RasterOutput, width: u32, height: u32) -> Result<()> {
    let expected = usize::try_from(width)
        .ok()
        .and_then(|width| {
            usize::try_from(height)
                .ok()
                .and_then(|height| width.checked_mul(height))
        })
        .ok_or_else(|| {
            PotError::new(
                ErrorCode::LimitExceeded,
                "render dimensions exceed platform limits",
            )
        })?;
    if output.linear_rgba.len() != expected
        || output.ids.len() != expected
        || output.depths.len() != expected
        || output.elements.len() != expected
        || output.normals.len() != expected
        || output.albedo.len() != expected
        || output.emission.len() != expected
        || output.ambient_occlusion.len() != expected
        || expected.checked_mul(4) != Some(output.rgba.len())
    {
        return Err(PotError::new(
            ErrorCode::RenderFailed,
            "render pass dimensions are invalid",
        ));
    }
    Ok(())
}

fn map_rgb(image: &Image, mut transform: impl FnMut(f32) -> f32) -> Image {
    Image {
        width: image.width,
        height: image.height,
        pixels: image
            .pixels
            .iter()
            .map(|pixel| {
                [
                    transform(pixel[0]),
                    transform(pixel[1]),
                    transform(pixel[2]),
                    pixel[3],
                ]
            })
            .collect(),
    }
}

fn map_alpha(image: &Image, alpha: f32) -> Image {
    Image {
        width: image.width,
        height: image.height,
        pixels: image
            .pixels
            .iter()
            .map(|pixel| [pixel[0], pixel[1], pixel[2], alpha])
            .collect(),
    }
}

fn set_alpha(node: &GraphNode, image: &Image, inputs: &BTreeMap<String, Image>) -> Result<Image> {
    let Some(alpha_image) = inputs.get("Alpha").or_else(|| inputs.get("alpha")) else {
        let alpha = scalar_default_value(node, &["Alpha"], &["alpha", "value"], 1.0)?;
        return Ok(map_alpha(image, alpha));
    };
    ensure_same_dimensions(image, alpha_image)?;
    Image::new(
        image.width,
        image.height,
        image
            .pixels
            .iter()
            .zip(&alpha_image.pixels)
            .map(|(pixel, alpha)| [pixel[0], pixel[1], pixel[2], alpha[0]])
            .collect(),
    )
}

fn map_color_balance(image: &Image, lift: [f32; 3], gamma: [f32; 3], gain: [f32; 3]) -> Image {
    Image {
        width: image.width,
        height: image.height,
        pixels: image
            .pixels
            .iter()
            .map(|pixel| {
                [0, 1, 2]
                    .map(|channel| {
                        ((pixel[channel] + lift[channel] - 1.0)
                            .max(0.0)
                            .powf(1.0 / gamma[channel]))
                            * gain[channel]
                    })
                    .pipe_alpha(pixel[3])
            })
            .collect(),
    }
}

trait AlphaChannel {
    fn pipe_alpha(self, alpha: f32) -> [f32; 4];
}
impl AlphaChannel for [f32; 3] {
    fn pipe_alpha(self, alpha: f32) -> [f32; 4] {
        [self[0], self[1], self[2], alpha]
    }
}

fn map_hue_sat(image: &Image, hue_offset: f32, saturation_scale: f32, value_scale: f32) -> Image {
    Image {
        width: image.width,
        height: image.height,
        pixels: image
            .pixels
            .iter()
            .map(|pixel| {
                let (hue, saturation, value) = rgb_to_hsv(pixel[0], pixel[1], pixel[2]);
                let (red, green, blue) = hsv_to_rgb(
                    (hue + hue_offset - 0.5).rem_euclid(1.0),
                    (saturation * saturation_scale).clamp(0.0, 2.0),
                    (value * value_scale).max(0.0),
                );
                [red, green, blue, pixel[3]]
            })
            .collect(),
    }
}

fn rgb_to_hsv(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    let maximum = r.max(g).max(b);
    let minimum = r.min(g).min(b);
    let delta = maximum - minimum;
    let hue = if crate::float::equal_f32(delta, 0.0) {
        0.0
    } else if crate::float::equal_f32(maximum, r) {
        ((g - b) / delta).rem_euclid(6.0) / 6.0
    } else if crate::float::equal_f32(maximum, g) {
        ((b - r) / delta + 2.0) / 6.0
    } else {
        ((r - g) / delta + 4.0) / 6.0
    };
    (
        hue,
        if crate::float::equal_f32(maximum, 0.0) {
            0.0
        } else {
            delta / maximum
        },
        maximum,
    )
}

fn hsv_to_rgb(hue: f32, saturation: f32, value: f32) -> (f32, f32, f32) {
    let sector = hue * 6.0;
    let index = sector.floor() as i32;
    let fraction = sector - index as f32;
    let low = value * (1.0 - saturation);
    let descending = value * (1.0 - fraction * saturation);
    let ascending = value * (1.0 - (1.0 - fraction) * saturation);
    match index.rem_euclid(6) {
        0 => (value, ascending, low),
        1 => (descending, value, low),
        2 => (low, value, ascending),
        3 => (low, descending, value),
        4 => (ascending, low, value),
        _ => (value, low, descending),
    }
}

fn curve_property(node: &GraphNode) -> Result<Vec<[f64; 2]>> {
    let Some(value) = node
        .properties
        .get("curve")
        .or_else(|| node.inputs.get("curve"))
    else {
        return Ok(Vec::new());
    };
    let points = value
        .as_array()
        .ok_or_else(|| PotError::invalid_argument("compositor curve must be an array"))?;
    let mut curve = Vec::with_capacity(points.len());
    for point in points {
        let pair = point
            .as_array()
            .filter(|pair| pair.len() == 2)
            .ok_or_else(|| {
                PotError::invalid_argument("compositor curve points must contain two values")
            })?;
        let x = pair[0]
            .as_f64()
            .filter(|value| value.is_finite())
            .ok_or_else(|| PotError::invalid_argument("compositor curve x must be finite"))?;
        let y = pair[1]
            .as_f64()
            .filter(|value| value.is_finite())
            .ok_or_else(|| PotError::invalid_argument("compositor curve y must be finite"))?;
        if curve
            .last()
            .is_some_and(|previous: &[f64; 2]| x <= previous[0])
        {
            return Err(PotError::invalid_argument(
                "compositor curve x values must be strictly increasing",
            ));
        }
        curve.push([x, y]);
    }
    Ok(curve)
}

fn color_property(node: &GraphNode, key: &str, default: [f32; 3]) -> Result<[f32; 3]> {
    let aliases: &[&str] = match key {
        "color1" => &["color1", "Color1"],
        "color2" => &["color2", "Color2"],
        "foreground" => &["foreground", "Image", "Foreground"],
        "background" => &["background", "Image_001", "Background"],
        "lift" => &["lift", "Lift"],
        "gamma" => &["gamma", "Gamma"],
        "gain" => &["gain", "Gain"],
        _ => &[key],
    };
    let Some(value) = aliases.iter().find_map(|alias| {
        node.properties
            .get(*alias)
            .or_else(|| node.inputs.get(*alias))
    }) else {
        return Ok(default);
    };
    let values = value
        .as_array()
        .filter(|values| values.len() >= 3)
        .ok_or_else(|| {
            PotError::invalid_argument("compositor color property must have RGB values")
        })?;
    let mut color = [0.0; 3];
    for (index, channel) in color.iter_mut().enumerate() {
        let value = values[index]
            .as_f64()
            .filter(|value| value.is_finite())
            .ok_or_else(|| {
                PotError::invalid_argument("compositor color channels must be finite")
            })?;
        *channel = value as f32;
        if !channel.is_finite() {
            return Err(PotError::invalid_argument(
                "compositor color channel exceeds f32 range",
            ));
        }
    }
    Ok(color)
}

fn color_image(node: &GraphNode, key: &str, base: &Image, default: [f32; 3]) -> Result<Image> {
    let color = color_property(node, key, default)?;
    Image::filled(base.width, base.height, [color[0], color[1], color[2], 1.0])
}

fn ensure_same_dimensions(first: &Image, second: &Image) -> Result<()> {
    if first.width != second.width || first.height != second.height {
        return Err(PotError::invalid_argument(
            "compositor image inputs must have matching dimensions",
        ));
    }
    Ok(())
}

fn is_type(node: &GraphNode, expected: &str) -> bool {
    node.node_type == expected
}

fn property_text<'a>(node: &'a GraphNode, names: &[&str]) -> Result<Option<&'a str>> {
    for name in names {
        if let Some(value) = node.properties.get(*name) {
            return value.as_str().map(Some).ok_or_else(|| {
                PotError::invalid_argument("compositor text property must be a string")
            });
        }
    }
    Ok(None)
}

fn property_f64(node: &GraphNode, names: &[&str], default: f64) -> Result<f64> {
    for name in names {
        if let Some(value) = node.properties.get(*name) {
            return value
                .as_f64()
                .filter(|value| value.is_finite())
                .ok_or_else(|| {
                    PotError::invalid_argument("compositor numeric property must be finite")
                });
        }
    }
    Ok(default)
}

fn property_u32(node: &GraphNode, names: &[&str], default: u32) -> Result<u32> {
    for name in names {
        if let Some(value) = node.properties.get(*name) {
            let integer = value.as_u64().ok_or_else(|| {
                PotError::invalid_argument("compositor integer property must be unsigned")
            })?;
            return u32::try_from(integer).map_err(|_| {
                PotError::invalid_argument("compositor integer property exceeds u32")
            });
        }
    }
    Ok(default)
}

fn property_i32(node: &GraphNode, names: &[&str], default: i32) -> Result<i32> {
    for name in names {
        if let Some(value) = node.properties.get(*name) {
            let integer = value.as_i64().ok_or_else(|| {
                PotError::invalid_argument("compositor offset must be an integer")
            })?;
            return i32::try_from(integer)
                .map_err(|_| PotError::invalid_argument("compositor offset exceeds i32"));
        }
    }
    Ok(default)
}
