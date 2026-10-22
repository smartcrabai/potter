#![expect(clippy::unwrap_used, reason = "integration test setup")]

use std::{cell::RefCell, collections::BTreeMap, f64::consts::FRAC_1_SQRT_2};

use glam::DVec3;
use potter_core::{
    graph::{GraphKind, GraphLink, GraphNode, NodeGroup},
    image::{ImageData, ImageInterpolation},
    model::{Id, Material, Registry, TextureRef},
    shader::{HitContext, HitUvLayers, evaluate_surface, height_gradient},
};
use serde_json::{Value, json};

/// Corners of the test triangle in the plane z = 0, so its normal is +Z. The hit is at its centre.
const POSITIONS: [DVec3; 3] = [DVec3::ZERO, DVec3::X, DVec3::Y];
const CENTRE: [f64; 3] = [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0];

/// UV = (x, y).
const RENDER: [[f64; 2]; 3] = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
/// UV = (-y, x): the render layer turned by 90 degrees.
const ROTATED: [[f64; 2]; 3] = [[0.0, 0.0], [0.0, 1.0], [-1.0, 0.0]];
/// UV = (2x, -y / 2): scaled, and mirrored in V.
const SCALED_MIRRORED: [[f64; 2]; 3] = [[0.0, 0.0], [2.0, 0.0], [0.0, -0.5]];
/// UV = (2y, x / 2): scaled and swapped.
const SWAPPED: [[f64; 2]; 3] = [[0.0, 0.0], [0.0, 0.5], [2.0, 0.0]];
/// One UV at every corner: the layer has no surface parametrisation.
const DEGENERATE: [[f64; 2]; 3] = [[0.5, 0.5]; 3];

/// Tangent-space normal (0.48, 0.6, 0.64) stored as the RGBA texel that encodes it.
const TEXEL: [f64; 4] = [0.74, 0.8, 0.82, 1.0];

/// The UV layers of the test mesh, owned so that a hit can borrow them.
struct Layers {
    names: Vec<String>,
    corners: Vec<[[f64; 2]; 3]>,
    render: Option<usize>,
}

impl Layers {
    fn new(layers: &[(&str, [[f64; 2]; 3])], render: &str) -> Self {
        Self {
            names: layers.iter().map(|(name, _)| (*name).to_owned()).collect(),
            corners: layers.iter().map(|(_, corners)| *corners).collect(),
            render: layers.iter().position(|(name, _)| *name == render),
        }
    }

    fn hit(&self) -> HitUvLayers<'_> {
        HitUvLayers {
            names: &self.names,
            corners: &self.corners,
            positions: POSITIONS,
            barycentric: CENTRE,
            render: self.render,
            available_mask: None,
        }
    }
}

fn assert_close(actual: DVec3, expected: DVec3) {
    assert!(
        (actual - expected).length() < 1.0e-9,
        "normal {actual:?} differs from {expected:?}"
    );
}

fn node(node_type: &str, properties: &[(&str, Value)], inputs: &[(&str, Value)]) -> GraphNode {
    let mut node = GraphNode::new(node_type);
    node.properties = properties
        .iter()
        .map(|(key, value)| ((*key).to_owned(), value.clone()))
        .collect();
    node.inputs = inputs
        .iter()
        .map(|(key, value)| ((*key).to_owned(), value.clone()))
        .collect();
    node
}

/// A material whose shader graph holds `nodes`, wired by `links` as (from node, output, to node,
/// input).
fn graph_material(
    nodes: &[(&str, GraphNode)],
    links: &[(&str, &str, &str, &str)],
) -> (Material, Registry<NodeGroup>) {
    let mut group = NodeGroup::new("Test", GraphKind::Shader);
    for (name, node) in nodes {
        group.nodes.insert(Id::new(*name).unwrap(), node.clone());
    }
    for (from, from_socket, to, to_socket) in links {
        group.links.push(GraphLink {
            from_node: Id::new(*from).unwrap(),
            from_socket: (*from_socket).to_owned(),
            to_node: Id::new(*to).unwrap(),
            to_socket: (*to_socket).to_owned(),
        });
    }
    let graph_id = Id::new("uv_layers_shader").unwrap();
    let mut groups = Registry::new();
    groups.insert(graph_id.clone(), group);
    let material = Material {
        node_tree: Some(graph_id),
        ..Material::default()
    };
    (material, groups)
}

/// A Bump whose Height is the U of a UV Map node reading `uv_map`; the Diffuse BSDF outputs the
/// bumped normal.
fn bump_material(uv_map: Option<&str>) -> (Material, Registry<NodeGroup>) {
    let properties: Vec<(&str, Value)> = uv_map
        .map(|layer| ("uv_map", json!(layer)))
        .into_iter()
        .collect();
    graph_material(
        &[
            (
                "out",
                node("OutputMaterial", &[("is_active_output", json!(true))], &[]),
            ),
            ("diffuse", node("ShaderNodeBsdfDiffuse", &[], &[])),
            (
                "bump",
                node("ShaderNodeBump", &[], &[("Distance", json!(1.0))]),
            ),
            ("uv", node("ShaderNodeUVMap", &properties, &[])),
        ],
        &[
            ("uv", "UV", "bump", "Height"),
            ("bump", "Normal", "diffuse", "Normal"),
            ("diffuse", "BSDF", "out", "Surface"),
        ],
    )
}

/// A Normal Map node decoding `TEXEL`, reading the layer `uv_map`; the Diffuse BSDF outputs it.
fn normal_map_material(uv_map: Option<&str>) -> (Material, Registry<NodeGroup>) {
    let properties: Vec<(&str, Value)> = uv_map
        .map(|layer| ("uv_map", json!(layer)))
        .into_iter()
        .collect();
    graph_material(
        &[
            (
                "out",
                node("OutputMaterial", &[("is_active_output", json!(true))], &[]),
            ),
            ("diffuse", node("ShaderNodeBsdfDiffuse", &[], &[])),
            (
                "normal",
                node(
                    "ShaderNodeNormalMap",
                    &properties,
                    &[("Color", json!(TEXEL))],
                ),
            ),
        ],
        &[
            ("normal", "Normal", "diffuse", "Normal"),
            ("diffuse", "BSDF", "out", "Surface"),
        ],
    )
}

/// A material whose normal texture is `TEXEL` on `uv_layer`.
fn texture_material(uv_layer: Option<&str>) -> (Material, BTreeMap<String, ImageData>) {
    let material = Material {
        normal_texture: Some(TextureRef {
            image: Id::new("normal_image").unwrap(),
            uv_layer: uv_layer.map(str::to_owned),
            interpolation: ImageInterpolation::Closest,
            extension: potter_core::image::ImageExtension::Repeat,
        }),
        ..Material::default()
    };
    let image = ImageData {
        colour: potter_core::image::ColourCache::default(),
        storage: potter_core::image::ImageStorage::default(),
        width: 1,
        height: 1,
        pixels: vec![TEXEL],
        tiles: BTreeMap::new(),
        interpolation: ImageInterpolation::Closest,
        srgb: false,
        udim: false,
    };
    (
        material,
        BTreeMap::from([("normal_image".to_owned(), image)]),
    )
}

fn surface_normal(material: &Material, groups: &Registry<NodeGroup>, layers: &Layers) -> DVec3 {
    let context = HitContext::new(groups).with_uv_layers(layers.hit());
    evaluate_surface(material, &context)
        .unwrap()
        .dominant()
        .unwrap()
        .normal
}

#[test]
fn bump_height_from_the_render_layer_tilts_the_normal_by_its_gradient() {
    // Height = u = x, so the surface gradient is (1, 0, 0) and the normal is (0, 0, 1) - (1, 0, 0).
    let layers = Layers::new(&[("UVMap", RENDER)], "UVMap");
    let (material, groups) = bump_material(Some("UVMap"));
    assert_close(
        surface_normal(&material, &groups, &layers),
        DVec3::new(-FRAC_1_SQRT_2, 0.0, FRAC_1_SQRT_2),
    );
}

#[test]
fn bump_height_from_a_swapped_layer_follows_its_physical_derivative() {
    // Render = UVMap. Height = Warp U = 2y, so the gradient is (0, 2, 0) and the normal is
    // (0, -2, 1) / sqrt(5).
    let layers = Layers::new(&[("UVMap", RENDER), ("Warp", SWAPPED)], "UVMap");
    let (material, groups) = bump_material(Some("Warp"));
    assert_close(
        surface_normal(&material, &groups, &layers),
        DVec3::new(0.0, -2.0, 1.0) / 5.0_f64.sqrt(),
    );
}

#[test]
fn bump_height_from_a_scaled_mirrored_layer_follows_its_physical_derivative() {
    // Height = Scaled U = 2x, so the gradient is (2, 0, 0) and the normal is (-2, 0, 1) / sqrt(5).
    let layers = Layers::new(&[("UVMap", RENDER), ("Scaled", SCALED_MIRRORED)], "UVMap");
    let (material, groups) = bump_material(Some("Scaled"));
    assert_close(
        surface_normal(&material, &groups, &layers),
        DVec3::new(-2.0, 0.0, 1.0) / 5.0_f64.sqrt(),
    );
}

#[test]
fn naming_the_render_layer_is_the_same_as_the_default() {
    // Render = Warp. The physical gradient does not depend on which layer the step follows.
    let layers = Layers::new(&[("UVMap", RENDER), ("Warp", SWAPPED)], "Warp");
    let (material, groups) = bump_material(Some("Warp"));
    let named = surface_normal(&material, &groups, &layers);
    let (material, groups) = bump_material(Some(""));
    let empty = surface_normal(&material, &groups, &layers);
    let (material, groups) = bump_material(None);
    let unnamed = surface_normal(&material, &groups, &layers);
    assert_close(named, DVec3::new(0.0, -2.0, 1.0) / 5.0_f64.sqrt());
    assert_eq!(named, empty);
    assert_eq!(empty, unnamed);
}

#[test]
fn empty_layer_name_is_the_render_layer_and_a_missing_name_warns() {
    let layers = Layers::new(&[("UVMap", RENDER)], "UVMap");
    let expected = DVec3::new(-FRAC_1_SQRT_2, 0.0, FRAC_1_SQRT_2);
    let warnings = RefCell::new(Vec::<Value>::new());

    let (material, groups) = bump_material(Some(""));
    let context = HitContext::new(&groups)
        .with_uv_layers(layers.hit())
        .with_warnings(&warnings);
    assert_close(
        evaluate_surface(&material, &context)
            .unwrap()
            .dominant()
            .unwrap()
            .normal,
        expected,
    );
    assert!(warnings.borrow().is_empty());

    let (material, groups) = bump_material(Some("Missing"));
    let context = HitContext::new(&groups)
        .with_uv_layers(layers.hit())
        .with_warnings(&warnings);
    assert_close(
        evaluate_surface(&material, &context)
            .unwrap()
            .dominant()
            .unwrap()
            .normal,
        expected,
    );
    assert_eq!(warnings.borrow().len(), 1);
    assert_eq!(warnings.borrow()[0]["code"], json!("uv.layer_missing"));
    assert_eq!(warnings.borrow()[0]["layer"], json!("Missing"));
}

#[test]
fn normal_texture_uses_the_frame_of_its_own_layer() {
    // Decoded normal (0.48, 0.6, 0.64) in the frame of each layer.
    // Render: T = (1, 0, 0), B = (0, 1, 0).
    // ROTATED: T = (0, -1, 0), B = (1, 0, 0), since dP/du = -y and dP/dv = x.
    // SCALED_MIRRORED: T = (1, 0, 0), B = (0, -1, 0), since dP/dv = -2y.
    let layers = Layers::new(
        &[
            ("UVMap", RENDER),
            ("UVMap.001", ROTATED),
            ("UVMap.002", SCALED_MIRRORED),
        ],
        "UVMap",
    );
    let groups = Registry::new();
    let cases = [
        (None, DVec3::new(0.48, 0.6, 0.64)),
        (Some("UVMap.001"), DVec3::new(0.6, -0.48, 0.64)),
        (Some("UVMap.002"), DVec3::new(0.48, -0.6, 0.64)),
    ];
    for (uv_layer, expected) in cases {
        let (material, images) = texture_material(uv_layer);
        let context = HitContext::new(&groups)
            .with_images(&images)
            .with_uv_layers(layers.hit());
        assert_close(
            evaluate_surface(&material, &context)
                .unwrap()
                .dominant()
                .unwrap()
                .normal,
            expected,
        );
    }
}

#[test]
fn normal_texture_empty_layer_is_render_and_missing_layer_warns() {
    let layers = Layers::new(&[("UVMap", RENDER), ("UVMap.001", ROTATED)], "UVMap");
    let groups = Registry::new();
    let render = DVec3::new(0.48, 0.6, 0.64);
    let warnings = RefCell::new(Vec::<Value>::new());
    for uv_layer in [None, Some("")] {
        let (material, images) = texture_material(uv_layer);
        let context = HitContext::new(&groups)
            .with_images(&images)
            .with_uv_layers(layers.hit())
            .with_warnings(&warnings);
        assert_close(
            evaluate_surface(&material, &context)
                .unwrap()
                .dominant()
                .unwrap()
                .normal,
            render,
        );
    }
    assert!(warnings.borrow().is_empty());

    let (material, images) = texture_material(Some("Missing"));
    let context = HitContext::new(&groups)
        .with_images(&images)
        .with_uv_layers(layers.hit())
        .with_warnings(&warnings);
    assert_close(
        evaluate_surface(&material, &context)
            .unwrap()
            .dominant()
            .unwrap()
            .normal,
        render,
    );
    assert_eq!(warnings.borrow().len(), 1);
    assert_eq!(warnings.borrow()[0]["layer"], json!("Missing"));
}

#[test]
fn normal_map_node_frame_comes_from_its_uv_map_property() {
    let layers = Layers::new(&[("UVMap", RENDER), ("UVMap.001", ROTATED)], "UVMap");
    let warnings = RefCell::new(Vec::<Value>::new());
    let cases = [
        (None, DVec3::new(0.48, -0.6, 0.64)),
        (Some(""), DVec3::new(0.48, -0.6, 0.64)),
        (Some("UVMap.001"), DVec3::new(-0.6, -0.48, 0.64)),
    ];
    for (uv_map, expected) in cases {
        let (material, groups) = normal_map_material(uv_map);
        let context = HitContext::new(&groups)
            .with_uv_layers(layers.hit())
            .with_warnings(&warnings);
        let actual = evaluate_surface(&material, &context)
            .unwrap()
            .dominant()
            .unwrap()
            .normal;
        assert_close(actual, expected);
        if uv_map == Some("UVMap.001") {
            // Blender 5.2.2 bake: canonical UV values converted to Blender V-up, with the selected
            // rotated layer, yield this world normal for the same encoded tangent-space texel.
            let blender = DVec3::new(-0.6, -0.4824, 0.6392);
            assert!(
                (actual - blender).length() < 3.0e-3,
                "normal {actual:?} differs from Blender bake {blender:?}"
            );
        }
    }
    assert!(warnings.borrow().is_empty());
}

#[test]
fn imported_normal_map_matches_direct_texture_with_unrelated_graph_nodes() {
    let layers = Layers::new(&[("UVMap", RENDER)], "UVMap");
    let (direct_material, images) = texture_material(None);
    let direct_groups = Registry::new();
    let direct_context = HitContext::new(&direct_groups)
        .with_images(&images)
        .with_uv_layers(layers.hit());
    let expected = evaluate_surface(&direct_material, &direct_context)
        .unwrap()
        .dominant()
        .unwrap()
        .normal;

    let image_properties = [
        ("image", json!("normal_image")),
        ("interpolation", json!("closest")),
    ];
    let (material, groups) = graph_material(
        &[
            (
                "out",
                node("OutputMaterial", &[("is_active_output", json!(true))], &[]),
            ),
            ("principled", node("ShaderNodeBsdfPrincipled", &[], &[])),
            (
                "normal",
                node(
                    "ShaderNodeNormalMap",
                    &[("potter_gltf_imported_normal_map", json!(true))],
                    &[],
                ),
            ),
            (
                "normal_image",
                node("ShaderNodeTexImage", &image_properties, &[]),
            ),
            (
                "emission_image",
                node("ShaderNodeTexImage", &image_properties, &[]),
            ),
            (
                "emission_mapping",
                node(
                    "ShaderNodeMapping",
                    &[],
                    &[("Vector", json!([0.25, 0.75, 0.0]))],
                ),
            ),
        ],
        &[
            ("normal_image", "Color", "normal", "Color"),
            ("normal", "Normal", "principled", "Normal"),
            ("emission_mapping", "Vector", "emission_image", "Vector"),
            ("emission_image", "Color", "principled", "Emission Color"),
            ("principled", "BSDF", "out", "Surface"),
        ],
    );
    let context = HitContext::new(&groups)
        .with_images(&images)
        .with_uv_layers(layers.hit());
    assert_close(
        evaluate_surface(&material, &context)
            .unwrap()
            .dominant()
            .unwrap()
            .normal,
        expected,
    );
}

#[test]
fn height_gradient_shifts_each_layer_by_its_own_derivative() {
    let layers = Layers::new(&[("UVMap", RENDER), ("Warp", SWAPPED)], "UVMap");
    let groups = Registry::new();
    let context = HitContext::new(&groups).with_uv_layers(layers.hit());
    // Height = render U = x: gradient (1, 0, 0).
    let render_u = height_gradient(&context, DVec3::Z, |sample| {
        Ok(sample.uv_layers.unwrap().uv(0).unwrap()[0])
    })
    .unwrap();
    assert_close(render_u, DVec3::X);
    // Height = Warp U = 2y: gradient (0, 2, 0).
    let warp_u = height_gradient(&context, DVec3::Z, |sample| {
        Ok(sample.uv_layers.unwrap().uv(1).unwrap()[0])
    })
    .unwrap();
    assert_close(warp_u, DVec3::new(0.0, 2.0, 0.0));
    // Height = render V = y, read from the hit's render UV: gradient (0, 1, 0).
    let render_v = height_gradient(&context, DVec3::Z, |sample| Ok(sample.uv[1])).unwrap();
    assert_close(render_v, DVec3::Y);
}

#[test]
fn degenerate_render_layer_leaves_uv_driven_height_flat() {
    // Every corner has UV (0.5, 0.5), so the height never changes across the triangle.
    let layers = Layers::new(&[("UVMap", DEGENERATE)], "UVMap");
    let (material, groups) = bump_material(Some("UVMap"));
    assert_close(surface_normal(&material, &groups, &layers), DVec3::Z);
}
