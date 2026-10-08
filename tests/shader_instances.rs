#![expect(clippy::unwrap_used, reason = "integration test setup")]

use std::collections::BTreeMap;

use glam::DVec3;
use potter::{
    error::ErrorCode,
    graph::{GraphKind, GraphLink, GraphNode, NodeGroup},
    image::{ImageData, ImageInterpolation, ImageTileData},
    model::{Id, Material, Registry, SceneDoc, TextureRef},
    ops::apply_batch,
    shader::{HitContext, evaluate_surface},
};
use serde_json::json;

fn add_node(
    group: &mut NodeGroup,
    id: &str,
    node_type: &str,
    inputs: &[(&str, serde_json::Value)],
) {
    let mut node = GraphNode::new(node_type);
    node.inputs = inputs
        .iter()
        .map(|(name, value)| ((*name).to_owned(), value.clone()))
        .collect();
    group.nodes.insert(Id::new(id).unwrap(), node);
}

fn link(group: &mut NodeGroup, from: &str, from_socket: &str, to: &str, to_socket: &str) {
    group.links.push(GraphLink {
        from_node: Id::new(from).unwrap(),
        from_socket: from_socket.to_owned(),
        to_node: Id::new(to).unwrap(),
        to_socket: to_socket.to_owned(),
    });
}

fn image_data(width: u32, height: u32, pixels: Vec<[f64; 4]>) -> ImageData {
    ImageData {
        width,
        height,
        pixels,
        tiles: BTreeMap::new(),
        interpolation: ImageInterpolation::Linear,
    }
}

fn texture_ref(image: &str, uv_map: Option<&str>) -> TextureRef {
    TextureRef {
        image: Id::new(image).unwrap(),
        uv_map: uv_map.map(str::to_owned),
        interpolation: ImageInterpolation::Closest,
    }
}

fn material_texture_images() -> BTreeMap<String, ImageData> {
    BTreeMap::from([
        (
            "base_image".to_owned(),
            image_data(2, 1, vec![[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 0.5]]),
        ),
        (
            "roughness_image".to_owned(),
            image_data(1, 1, vec![[0.25, 0.0, 0.0, 1.0]]),
        ),
        (
            "metallic_image".to_owned(),
            image_data(1, 1, vec![[0.4, 0.0, 0.0, 1.0]]),
        ),
        (
            "normal_image".to_owned(),
            image_data(1, 1, vec![[0.5, 0.5, 1.0, 1.0]]),
        ),
    ])
}

fn material_with_graph(graph: NodeGroup) -> (Material, Registry<NodeGroup>) {
    let graph_id = Id::new("shader_test").unwrap();
    let mut groups = Registry::new();
    groups.insert(graph_id.clone(), graph);
    let material = Material {
        name: "Test".to_owned(),
        base_color: [0.6, 0.6, 0.6, 1.0],
        metallic: 0.0,
        roughness: 0.8,
        emission_color: [0.0; 3],
        emission_strength: 0.0,
        transmission: 0.0,
        ior: 1.45,
        double_sided: false,
        node_tree: Some(graph_id),
        ..Material::default()
    };
    (material, groups)
}

#[test]
fn simple_pbr_material_evaluates_through_its_principled_graph() {
    let doc = SceneDoc::new("scene-id".to_owned());
    let applied = apply_batch(
        &doc,
        &json!({
            "schema_version": 1,
            "base_revision": 0,
            "operations": [{
                "op": "material.create",
                "id": "clay",
                "base_color": [0.25, 0.5, 0.75, 1.0],
                "metallic": 0.4,
                "roughness": 0.3
            }]
        }),
    )
    .unwrap();
    let material = applied
        .doc
        .materials
        .get(&Id::new("clay").unwrap())
        .unwrap();
    assert!(
        material.node_tree.is_some(),
        "simple PBR creates a shader node tree"
    );
    let context = HitContext::new(&applied.doc.node_groups);
    let graph_bsdf = evaluate_surface(material, &context).unwrap();
    let mut simple = material.clone();
    simple.node_tree = None;
    let simple_bsdf = evaluate_surface(&simple, &context).unwrap();
    assert_eq!(graph_bsdf, simple_bsdf);
    assert_eq!(graph_bsdf.base_color, [0.25, 0.5, 0.75, 1.0]);
    assert_eq!(graph_bsdf.metallic, 0.4);
    assert_eq!(graph_bsdf.roughness, 0.3);
}

#[test]
fn checker_texture_alternates_colors_across_uv_tiles() {
    let mut graph = NodeGroup::new("Checker", GraphKind::Shader);
    add_node(&mut graph, "out", "OutputMaterial", &[]);
    add_node(&mut graph, "principled", "ShaderNodeBsdfPrincipled", &[]);
    add_node(
        &mut graph,
        "checker",
        "ShaderNodeTexChecker",
        &[
            ("Color1", json!([1.0, 0.0, 0.0, 1.0])),
            ("Color2", json!([0.0, 0.0, 1.0, 1.0])),
            ("Scale", json!(2.0)),
        ],
    );
    link(&mut graph, "checker", "Color", "principled", "Base Color");
    link(&mut graph, "principled", "BSDF", "out", "Surface");
    let (material, groups) = material_with_graph(graph);
    let mut context = HitContext::new(&groups);
    context.uv = [0.1, 0.1];
    let first = evaluate_surface(&material, &context).unwrap();
    context.uv = [0.6, 0.1];
    let second = evaluate_surface(&material, &context).unwrap();
    assert_eq!(first.base_color, [1.0, 0.0, 0.0, 1.0]);
    assert_eq!(second.base_color, [0.0, 0.0, 1.0, 1.0]);
}

#[test]
fn mix_shader_at_half_factor_averages_principled_parameters() {
    let mut graph = NodeGroup::new("Mix", GraphKind::Shader);
    add_node(&mut graph, "out", "OutputMaterial", &[]);
    add_node(
        &mut graph,
        "red",
        "ShaderNodeBsdfPrincipled",
        &[
            ("Base Color", json!([1.0, 0.0, 0.0, 1.0])),
            ("Metallic", json!(0.2)),
        ],
    );
    add_node(
        &mut graph,
        "blue",
        "ShaderNodeBsdfPrincipled",
        &[
            ("Base Color", json!([0.0, 0.0, 1.0, 1.0])),
            ("Metallic", json!(0.8)),
        ],
    );
    add_node(
        &mut graph,
        "mix",
        "ShaderNodeMixShader",
        &[("Fac", json!(0.5))],
    );
    link(&mut graph, "red", "BSDF", "mix", "Shader");
    link(&mut graph, "blue", "BSDF", "mix", "Shader_001");
    link(&mut graph, "mix", "Shader", "out", "Surface");
    let (material, groups) = material_with_graph(graph);
    let context = HitContext::new(&groups);
    let bsdf = evaluate_surface(&material, &context).unwrap();
    assert_eq!(bsdf.base_color, [0.5, 0.0, 0.5, 1.0]);
    assert!((bsdf.metallic - 0.5).abs() < 1e-12);
}

#[test]
fn add_shader_combines_emission_and_surface_shaders() {
    let mut graph = NodeGroup::new("Add", GraphKind::Shader);
    add_node(&mut graph, "out", "OutputMaterial", &[]);
    add_node(
        &mut graph,
        "surface",
        "ShaderNodeBsdfPrincipled",
        &[("Base Color", json!([0.2, 0.0, 0.0, 1.0]))],
    );
    add_node(
        &mut graph,
        "emission",
        "ShaderNodeEmission",
        &[
            ("Color", json!([0.0, 0.3, 0.0, 1.0])),
            ("Strength", json!(2.0)),
        ],
    );
    add_node(&mut graph, "add", "ShaderNodeAddShader", &[]);
    link(&mut graph, "surface", "BSDF", "add", "Shader");
    link(&mut graph, "emission", "Emission", "add", "Shader_001");
    link(&mut graph, "add", "Shader", "out", "Surface");
    let (material, groups) = material_with_graph(graph);
    let context = HitContext::new(&groups);
    let bsdf = evaluate_surface(&material, &context).unwrap();
    assert_eq!(bsdf.base_color, [0.2, 0.3, 0.0, 1.0]);
    assert_eq!(bsdf.emission_color, [0.0, 0.3, 0.0]);
    assert_eq!(bsdf.emission_strength, 2.0);
}
#[test]
fn material_create_accepts_inline_shader_node_tree() {
    let doc = SceneDoc::new("scene-id".to_owned());
    let applied = apply_batch(
        &doc,
        &json!({
            "schema_version": 1,
            "base_revision": 0,
            "operations": [{
                "op": "material.create",
                "id": "graph_mat",
                "graph": {
                    "kind": "shader",
                    "nodes": {
                        "out": {"type": "OutputMaterial"},
                        "principled": {
                            "type": "ShaderNodeBsdfPrincipled",
                            "inputs": {"Base Color": [0.1, 0.2, 0.3, 1.0], "Metallic": 0.7}
                        }
                    },
                    "links": [{
                        "from_node": "principled", "from_socket": "BSDF",
                        "to_node": "out", "to_socket": "Surface"
                    }]
                }
            }]
        }),
    )
    .unwrap();
    let material = applied
        .doc
        .materials
        .get(&Id::new("graph_mat").unwrap())
        .unwrap();
    let context = HitContext::new(&applied.doc.node_groups);
    let bsdf = evaluate_surface(material, &context).unwrap();
    assert_eq!(bsdf.base_color, [0.1, 0.2, 0.3, 1.0]);
    assert_eq!(bsdf.metallic, 0.7);
}

#[test]
fn image_texture_samples_resolved_pixels_using_hit_uvs() {
    let mut graph = NodeGroup::new("Image", GraphKind::Shader);
    add_node(&mut graph, "out", "OutputMaterial", &[]);
    add_node(&mut graph, "principled", "ShaderNodeBsdfPrincipled", &[]);
    add_node(&mut graph, "image", "ShaderNodeTexImage", &[]);
    graph
        .nodes
        .get_mut(&Id::new("image").unwrap())
        .unwrap()
        .properties
        .insert("image".to_owned(), json!("test_image"));
    graph
        .nodes
        .get_mut(&Id::new("image").unwrap())
        .unwrap()
        .properties
        .insert("interpolation".to_owned(), json!("closest"));
    link(&mut graph, "image", "Color", "principled", "Base Color");
    link(&mut graph, "principled", "BSDF", "out", "Surface");
    let (material, groups) = material_with_graph(graph);
    let image_data = BTreeMap::from([(
        "test_image".to_owned(),
        ImageData {
            width: 2,
            height: 1,
            pixels: vec![[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0]],
            tiles: BTreeMap::from([(
                1002,
                ImageTileData {
                    width: 1,
                    height: 1,
                    pixels: vec![[0.0, 0.0, 1.0, 1.0]],
                },
            )]),
            interpolation: ImageInterpolation::Linear,
        },
    )]);
    let mut context = HitContext::new(&groups).with_images(&image_data);
    context.uv = [0.25, 0.5];
    let first = evaluate_surface(&material, &context).unwrap();
    assert_eq!(first.base_color, [1.0, 0.0, 0.0, 1.0]);
    context.uv = [0.75, 0.5];
    let closest = evaluate_surface(&material, &context).unwrap();
    assert_eq!(closest.base_color, [0.0, 1.0, 0.0, 1.0]);

    context.uv = [1.25, 0.5];
    let udim_tile = evaluate_surface(&material, &context).unwrap();
    assert_eq!(udim_tile.base_color, [0.0, 0.0, 1.0, 1.0]);
}

#[test]
fn image_texture_without_resolved_pixels_reports_unsupported_feature() {
    let mut graph = NodeGroup::new("Missing image", GraphKind::Shader);
    add_node(&mut graph, "out", "OutputMaterial", &[]);
    add_node(&mut graph, "principled", "ShaderNodeBsdfPrincipled", &[]);
    add_node(&mut graph, "image", "ShaderNodeTexImage", &[]);
    graph
        .nodes
        .get_mut(&Id::new("image").unwrap())
        .unwrap()
        .properties
        .insert("image".to_owned(), json!("test_image"));
    link(&mut graph, "image", "Color", "principled", "Base Color");
    link(&mut graph, "principled", "BSDF", "out", "Surface");
    let (material, groups) = material_with_graph(graph);
    let error = evaluate_surface(&material, &HitContext::new(&groups)).unwrap_err();
    assert_eq!(error.code, ErrorCode::UnsupportedFeature);
    assert_eq!(
        error.details["feature_id"],
        json!("shader.node.ShaderNodeTexImage.image_data")
    );
}

#[test]
fn material_volume_output_evaluates_volume_properties() {
    let mut graph = NodeGroup::new("Volume", GraphKind::Shader);
    add_node(&mut graph, "out", "OutputMaterial", &[]);
    add_node(
        &mut graph,
        "volume",
        "ShaderNodeVolumePrincipled",
        &[
            ("Density", json!(0.7)),
            ("Color", json!([0.2, 0.4, 0.6, 1.0])),
            ("Anisotropy", json!(0.35)),
        ],
    );
    link(&mut graph, "volume", "Volume", "out", "Volume");
    let (material, groups) = material_with_graph(graph);
    let params = evaluate_surface(&material, &HitContext::new(&groups)).unwrap();
    assert_eq!(params.volume_density, 0.7);
    assert_eq!(params.volume_color, [0.2, 0.4, 0.6]);
    assert_eq!(params.volume_anisotropy, 0.35);
    assert!(params.volume_scattering);
}
#[test]
fn material_texture_refs_sample_uv_and_override_pbr_fields_for_graphs() {
    let mut graph = NodeGroup::new("Simple PBR", GraphKind::Shader);
    add_node(&mut graph, "out", "OutputMaterial", &[]);
    add_node(&mut graph, "principled", "ShaderNodeBsdfPrincipled", &[]);
    graph
        .nodes
        .get_mut(&Id::new("principled").unwrap())
        .unwrap()
        .properties
        .insert("potter_simple_material".to_owned(), json!(true));
    link(&mut graph, "principled", "BSDF", "out", "Surface");
    let (mut material, groups) = material_with_graph(graph);
    let graph_id = material.node_tree.clone().unwrap();
    material.node_tree = None;
    material.base_color = [0.5, 0.25, 0.75, 1.0];
    material.metallic = 0.5;
    material.roughness = 0.8;
    material.base_color_texture = Some(texture_ref("base_image", Some("uv_map")));
    material.roughness_texture = Some(texture_ref("roughness_image", None));
    material.metallic_texture = Some(texture_ref("metallic_image", None));
    material.normal_texture = Some(texture_ref("normal_image", None));
    let image_data = material_texture_images();
    let uv_maps = [("uv_map", [0.75, 0.5])];
    let mut context = HitContext::new(&groups)
        .with_images(&image_data)
        .with_uv_maps(&uv_maps)
        .with_tangent_frame(DVec3::X, DVec3::Y);
    context.uv = [0.25, 0.5];
    let simple = evaluate_surface(&material, &context).unwrap();
    assert_eq!(simple.base_color, [0.0, 0.25, 0.0, 1.0]);
    assert_eq!(simple.alpha, 0.5);
    assert_eq!(simple.roughness, 0.2);
    assert_eq!(simple.metallic, 0.2);
    assert_eq!(simple.normal, DVec3::Z);

    material.node_tree = Some(graph_id);
    let graph_backed = evaluate_surface(&material, &context).unwrap();
    assert_eq!(graph_backed, simple);
    let no_images = HitContext::new(&groups).with_uv_maps(&uv_maps);
    let error = evaluate_surface(&material, &no_images).unwrap_err();
    assert_eq!(error.code, ErrorCode::UnsupportedFeature);
    assert_eq!(
        error.details["feature_id"],
        json!("shader.material_texture.image_data")
    );
    material.base_color_texture.as_mut().unwrap().uv_map = Some("missing_uv".to_owned());
    let error = evaluate_surface(&material, &context).unwrap_err();
    assert_eq!(error.code, ErrorCode::UnsupportedFeature);
    assert_eq!(
        error.details["feature_id"],
        json!("shader.material_texture.uv_map")
    );
}
#[test]
fn bump_node_perturbs_principled_normal_from_height_gradient() {
    let mut graph = NodeGroup::new("Bump", GraphKind::Shader);
    add_node(&mut graph, "out", "OutputMaterial", &[]);
    add_node(&mut graph, "principled", "ShaderNodeBsdfPrincipled", &[]);
    add_node(&mut graph, "bump", "ShaderNodeBump", &[]);
    add_node(&mut graph, "coordinates", "ShaderNodeTexCoord", &[]);
    add_node(
        &mut graph,
        "height",
        "ShaderNodeMath",
        &[("Value_001", json!(1.0))],
    );
    graph
        .nodes
        .get_mut(&Id::new("height").unwrap())
        .unwrap()
        .properties
        .insert("operation".to_owned(), json!("MULTIPLY"));
    link(&mut graph, "coordinates", "UV", "height", "Value");
    link(&mut graph, "height", "Value", "bump", "Height");
    link(&mut graph, "bump", "Normal", "principled", "Normal");
    link(&mut graph, "principled", "BSDF", "out", "Surface");
    let (material, groups) = material_with_graph(graph);
    let context = HitContext::new(&groups);
    let bsdf = evaluate_surface(&material, &context).unwrap();
    assert!(bsdf.normal.x < -0.5);
    assert!(bsdf.normal.z > 0.5);
}

#[test]
fn color_ramp_interpolates_across_configured_stops() {
    let mut graph = NodeGroup::new("Color ramp", GraphKind::Shader);
    add_node(&mut graph, "out", "OutputMaterial", &[]);
    add_node(&mut graph, "principled", "ShaderNodeBsdfPrincipled", &[]);
    add_node(
        &mut graph,
        "ramp",
        "ShaderNodeValToRGB",
        &[("Fac", json!(0.25))],
    );
    graph
        .nodes
        .get_mut(&Id::new("ramp").unwrap())
        .unwrap()
        .properties
        .insert(
            "elements".to_owned(),
            json!([
                {"position":0.0,"color":[1.0,0.0,0.0,1.0]},
                {"position":0.5,"color":[0.0,1.0,0.0,1.0]},
                {"position":1.0,"color":[0.0,0.0,1.0,1.0]}
            ]),
        );
    link(&mut graph, "ramp", "Color", "principled", "Base Color");
    link(&mut graph, "principled", "BSDF", "out", "Surface");
    let (material, groups) = material_with_graph(graph);
    let context = HitContext::new(&groups);
    let bsdf = evaluate_surface(&material, &context).unwrap();
    assert_eq!(bsdf.base_color, [0.5, 0.5, 0.0, 1.0]);
}

#[test]
fn tangent_normal_map_preserves_the_surface_normal() {
    let mut graph = NodeGroup::new("Normal map", GraphKind::Shader);
    add_node(&mut graph, "out", "OutputMaterial", &[]);
    add_node(&mut graph, "principled", "ShaderNodeBsdfPrincipled", &[]);
    add_node(&mut graph, "normal_map", "ShaderNodeNormalMap", &[]);
    link(&mut graph, "normal_map", "Normal", "principled", "Normal");
    link(&mut graph, "principled", "BSDF", "out", "Surface");
    let (material, groups) = material_with_graph(graph);
    let mut context = HitContext::new(&groups);
    context.normal = DVec3::X;
    let bsdf = evaluate_surface(&material, &context).unwrap();
    assert!((bsdf.normal.x - 1.0).abs() < 1.0e-12);
    assert!(bsdf.normal.y.abs() < 1.0e-12);
    assert!(bsdf.normal.z.abs() < 1.0e-12);
}

#[test]
fn hit_context_exposes_world_and_object_coordinates() {
    let groups = Registry::new();
    let mut context = HitContext::new(&groups);
    context.position = DVec3::new(1.0, 2.0, 3.0);
    context.object = DVec3::new(-1.0, -2.0, -3.0);
    let _attributes: BTreeMap<String, [f64; 4]> = BTreeMap::new();
    assert_eq!(context.position, DVec3::new(1.0, 2.0, 3.0));
    assert_eq!(context.object, DVec3::new(-1.0, -2.0, -3.0));
}
