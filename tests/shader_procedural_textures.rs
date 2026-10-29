#[path = "common/blender_checked.rs"]
mod blender_checked;
#[path = "common/native_oracle.rs"]
#[expect(
    dead_code,
    reason = "the shared oracle helper is used by multiple focused integration targets"
)]
mod native_oracle;
#[path = "common/process.rs"]
mod process;

use std::{collections::{BTreeMap, BTreeSet}, fs, path::Path};
use native_oracle::{
    TestResult, apply_operations, assert_close, checked, export, export_raw, import_blend,
    import_blend_allow_lossy, loss_features, losses, pot, read_scene,
};
use potter_core::{
    graph::{self, GraphKind, GraphLink, GraphNode, NodeGroup},
    image::{ImageData, ImageInterpolation, load_image_data, sample},
    model::{Id, Image, Material, Registry, SceneDoc},
    shader::{HitContext, evaluate_surface},
};
use serde_json::{Map, Value, json};
use tempfile::tempdir;

const PROBE_PIXEL_TOLERANCE: f64 = 1.0e-3;

#[test]
fn procedural_shader_textures_match_blender_and_roundtrip_without_losses() -> TestResult {
    let Some(blender) = blender_checked::blender_executable() else {
        return Ok(());
    };
    let root = tempdir()?;
    let fixtures = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/shader_procedural_textures");
    let blend = fixtures.join("procedural_textures.blend");
    let oracle_path = fixtures.join("blender_oracle.json");
    let oracle: Value = serde_json::from_slice(&fs::read(&oracle_path)?)?;
    assert_eq!(oracle["schema"], "shader-procedural-textures-oracle/1");
    assert_eq!(oracle["blender_version"], "5.2.2 LTS");
    assert_eq!(
        oracle["color_settings"],
        json!({"view_transform":"Raw", "look":"None", "exposure":0.0, "gamma":1.0})
    );
    assert_probe_coverage(&oracle)?;

    let (project, import_report) = import_blend(&blender, root.path(), "procedural", &blend)?;
    assert_lossless(&import_report, "procedural texture fixture import");
    let imported_scene = read_scene(&project)?;
    assert_imported_probes(&imported_scene, &oracle)?;
    assert_imported_gn_field(&imported_scene, &oracle)?;

    let exported = root.path().join("procedural-roundtrip.blend");
    let export_report = export(&blender, &project, "blend", &exported)?;
    assert_lossless(&export_report, "procedural texture fixture export");
    let (roundtrip, roundtrip_report) =
        import_blend(&blender, root.path(), "procedural_roundtrip", &exported)?;
    assert_lossless(&roundtrip_report, "procedural texture fixture re-import");
    assert_imported_probes(&read_scene(&roundtrip)?, &oracle)?;
    assert_imported_gn_field(&read_scene(&roundtrip)?, &oracle)?;

    for (engine, oracle_engine) in [("path", "cycles"), ("realtime", "eevee")] {
        let expected_resolution: [u32; 2] =
            serde_json::from_value(oracle["renders"][oracle_engine]["resolution"].clone())?;
        for (name, target_project) in [("imported", &project), ("roundtrip", &roundtrip)] {
            let image = render_project(
                target_project,
                root.path(),
                &format!("{name}_{oracle_engine}"),
                engine,
                expected_resolution,
            )?;
            compare_probe_pixels(&image, &oracle, oracle_engine)?;
        }
    }
    Ok(())
}

#[test]
fn procedural_world_texture_matches_cycles_and_eevee() -> TestResult {
    let Some(blender) = blender_checked::blender_executable() else {
        return Ok(());
    };
    let root = tempdir()?;
    let fixtures = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/shader_procedural_textures");
    let oracle: Value =
        serde_json::from_slice(&fs::read(fixtures.join("blender_oracle.json"))?)?;
    let blend = fixtures.join("world_procedural.blend");
    let (project, import_report) =
        import_blend(&blender, root.path(), "procedural_world", &blend)?;
    assert_lossless(&import_report, "procedural world texture import");

    for (engine, oracle_engine) in [("path", "cycles"), ("realtime", "eevee")] {
        let resolution: [u32; 2] =
            serde_json::from_value(oracle["world_render"]["renders"][oracle_engine]["resolution"].clone())?;
        let image = render_project(
            &project,
            root.path(),
            &format!("world_{oracle_engine}"),
            engine,
            resolution,
        )?;
        compare_world_pixels(&image, &oracle, oracle_engine)?;
    }
    Ok(())
}

#[test]
fn shader_wave_refuses_nondefault_texture_mapping() -> TestResult {
    let mut graph = NodeGroup::new("Mapped Wave", GraphKind::Shader);
    let output = Id::new("output")?;
    let emission = Id::new("emission")?;
    let wave = Id::new("wave")?;
    let mut wave_node = GraphNode::new("ShaderNodeTexWave");
    wave_node.properties.insert(
        "texture_mapping".to_owned(),
        json!({"vector_type":"POINT", "scale":[2.0, 1.0, 1.0]}),
    );
    graph.nodes.insert(output.clone(), GraphNode::new("ShaderNodeOutputMaterial"));
    graph.nodes.insert(emission.clone(), GraphNode::new("ShaderNodeEmission"));
    graph.nodes.insert(wave.clone(), wave_node);
    graph.links = vec![
        GraphLink {
            from_node: wave,
            from_socket: "Color".to_owned(),
            to_node: emission.clone(),
            to_socket: "Color".to_owned(),
        },
        GraphLink {
            from_node: emission,
            from_socket: "Emission".to_owned(),
            to_node: output,
            to_socket: "Surface".to_owned(),
        },
    ];

    let graph_id = Id::new("mapped_wave_graph")?;
    let mut groups = Registry::default();
    groups.insert(graph_id.clone(), graph);
    let material = Material {
        node_tree: Some(graph_id),
        ..Material::default()
    };
    let mut context = HitContext::new(&groups);
    context.path_traced = true;
    let error = evaluate_surface(&material, &context)
        .err()
        .ok_or("nondefault procedural texture mapping was accepted")?;
    assert_eq!(error.code, ErrorCode::UnsupportedFeature);
    assert_eq!(
        error.details["feature_id"],
        "shader.node.ShaderNodeTexWave.texture_mapping"
    );
    Ok(())
}

#[test]
fn shader_procedural_textures_reject_unknown_and_inactive_inputs() -> TestResult {
    let mut unknown = NodeGroup::new("Unknown Texture Input", GraphKind::Shader);
    let output = Id::new("output")?;
    let emission = Id::new("emission")?;
    let texture = Id::new("texture")?;
    let mut white_noise = GraphNode::new("ShaderNodeTexWhiteNoise");
    white_noise
        .inputs
        .insert("Unexpected".to_owned(), json!(0.5));
    unknown
        .nodes
        .insert(output.clone(), GraphNode::new("ShaderNodeOutputMaterial"));
    unknown
        .nodes
        .insert(emission.clone(), GraphNode::new("ShaderNodeEmission"));
    unknown.nodes.insert(texture.clone(), white_noise);
    unknown.links = vec![
        GraphLink {
            from_node: texture,
            from_socket: "Color".to_owned(),
            to_node: emission.clone(),
            to_socket: "Color".to_owned(),
        },
        GraphLink {
            from_node: emission,
            from_socket: "Emission".to_owned(),
            to_node: output,
            to_socket: "Surface".to_owned(),
        },
    ];
    let (code, message) = shader_surface_error(unknown)?;
    assert_eq!(code, ErrorCode::InvalidOperation);
    assert!(message.contains("unknown input socket `Unexpected`"));

    let mut inactive = NodeGroup::new("Inactive White Noise Input", GraphKind::Shader);
    let output = Id::new("output")?;
    let emission = Id::new("emission")?;
    let wave = Id::new("wave")?;
    let white_noise_id = Id::new("white_noise")?;
    let mut white_noise = GraphNode::new("ShaderNodeTexWhiteNoise");
    white_noise
        .properties
        .insert("noise_dimensions".to_owned(), json!("2D"));
    white_noise
        .inputs
        .insert("Vector".to_owned(), json!([0.1, 0.2, 0.3]));
    white_noise.inputs.insert("W".to_owned(), json!(0.73));
    let mut wave_node = GraphNode::new("ShaderNodeTexWave");
    wave_node
        .inputs
        .insert("Vector".to_owned(), json!([0.2, 0.4, 0.6]));
    inactive
        .nodes
        .insert(output.clone(), GraphNode::new("ShaderNodeOutputMaterial"));
    inactive
        .nodes
        .insert(emission.clone(), GraphNode::new("ShaderNodeEmission"));
    inactive.nodes.insert(wave.clone(), wave_node);
    inactive
        .nodes
        .insert(white_noise_id.clone(), white_noise);
    inactive.links = vec![
        GraphLink {
            from_node: wave,
            from_socket: "Fac".to_owned(),
            to_node: white_noise_id.clone(),
            to_socket: "W".to_owned(),
        },
        GraphLink {
            from_node: white_noise_id,
            from_socket: "Color".to_owned(),
            to_node: emission.clone(),
            to_socket: "Color".to_owned(),
        },
        GraphLink {
            from_node: emission,
            from_socket: "Emission".to_owned(),
            to_node: output,
            to_socket: "Surface".to_owned(),
        },
    ];
    let (code, message) = shader_surface_error(inactive)?;
    assert_eq!(code, ErrorCode::InvalidOperation);
    assert!(message.contains("input socket `W` is unavailable"));
    Ok(())
}

fn shader_surface_error(graph: NodeGroup) -> TestResult<(ErrorCode, String)> {
    let graph_id = Id::new("graph")?;
    let mut groups = Registry::default();
    groups.insert(graph_id.clone(), graph);
    let material = Material {
        node_tree: Some(graph_id),
        ..Material::default()
    };
    let error = evaluate_surface(&material, &HitContext::new(&groups))
        .err()
        .ok_or("invalid procedural texture input was accepted")?;
    Ok((error.code, error.to_string()))
}

#[test]
fn nondefault_blend_texture_mapping_refusal_roundtrips_its_nested_property() -> TestResult {
    let Some(blender) = blender_checked::blender_executable() else {
        return Ok(());
    };
    let root = tempdir()?;
    let fixtures = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/shader_procedural_textures");
    let blend = fixtures.join("mapped_texture_mapping.blend");
    let (project, import_report) =
        import_blend_allow_lossy(&blender, root.path(), "mapped_texture", &blend)?;
    assert_exact_mapping_refusal(&import_report);
    assert_mapped_wave_property(&read_scene(&project)?)?;

    let exported = root.path().join("mapped_texture_roundtrip.blend");
    let export_output = export_raw(&blender, &project, "blend", &exported, true)?;
    assert!(
        export_output.status.success(),
        "mapped texture export failed: {}",
        String::from_utf8_lossy(&export_output.stdout)
    );
    let export_report: Value = serde_json::from_slice(&export_output.stdout)?;
    assert_no_unexpected_mapping_losses(&export_report);

    let (roundtrip, reimport_report) =
        import_blend_allow_lossy(&blender, root.path(), "mapped_texture_roundtrip", &exported)?;
    assert_exact_mapping_refusal(&reimport_report);
    assert_mapped_wave_property(&read_scene(&roundtrip)?)?;
    Ok(())
}

fn assert_exact_mapping_refusal(report: &Value) {
    assert_eq!(
        loss_features(report),
        vec!["shader.node.ShaderNodeTexWave.texture_mapping".to_owned()],
        "unexpected losses in mapped texture report: {report}"
    );
}

fn assert_no_unexpected_mapping_losses(report: &Value) {
    let features = loss_features(report);
    assert!(
        features.is_empty()
            || features == vec!["shader.node.ShaderNodeTexWave.texture_mapping".to_owned()],
        "unexpected export losses for mapped texture: {report}"
    );
}

fn assert_mapped_wave_property(scene: &Value) -> TestResult {
    let (_, _, node) = find_node(scene, "Mapped Wave")?;
    assert_eq!(
        node["properties"]["texture_mapping"]["scale"],
        json!([2.0, 1.0, 1.0])
    );
    Ok(())
}

fn assert_lossless(report: &Value, label: &str) {
    assert!(losses(report).is_empty(), "{label} lost data: {report}");
}

fn assert_probe_coverage(oracle: &Value) -> TestResult {
    let probes = oracle["probes"]
        .as_array()
        .ok_or("oracle probes are missing")?;
    assert!(!probes.is_empty(), "oracle produced no probe tiles");
    assert_eq!(oracle["renders"]["cycles"]["pixels"].as_array().map(Vec::len), Some(probes.len()));
    assert_eq!(oracle["renders"]["eevee"]["pixels"].as_array().map(Vec::len), Some(probes.len()));

    let expected_nodes = BTreeSet::from([
        "ShaderNodeTexWave",
        "ShaderNodeTexMagic",
        "ShaderNodeTexBrick",
        "ShaderNodeTexWhiteNoise",
        "ShaderNodeTexGabor",
    ]);
    let expected_outputs = BTreeMap::from([
        ("ShaderNodeTexWave", BTreeSet::from(["Color", "Fac"])),
        ("ShaderNodeTexMagic", BTreeSet::from(["Color", "Fac"])),
        ("ShaderNodeTexBrick", BTreeSet::from(["Color", "Fac"])),
        ("ShaderNodeTexWhiteNoise", BTreeSet::from(["Color", "Value"])),
        (
            "ShaderNodeTexGabor",
            BTreeSet::from(["Value", "Phase", "Intensity"]),
        ),
    ]);
    let mut node_types = BTreeSet::new();
    let mut outputs = BTreeMap::<&str, BTreeSet<&str>>::new();
    let mut coordinates = BTreeMap::<&str, BTreeSet<&str>>::new();
    let mut waves = BTreeSet::<(String, String, String)>::new();
    let mut white_noise_dimensions = BTreeSet::new();
    let mut gabor_dimensions = BTreeSet::new();
    let mut magic_depths = BTreeSet::new();
    let mut names = BTreeSet::new();
    for probe in probes {
        let id = probe["id"].as_str().ok_or("probe id is missing")?;
        assert!(names.insert(id), "duplicate probe {id}");
        let node = probe["node_type"].as_str().ok_or("probe node type is missing")?;
        let output = probe["output"].as_str().ok_or("probe output is missing")?;
        let coordinate = probe["coordinate_source"]
            .as_str()
            .ok_or("probe coordinate source is missing")?;
        node_types.insert(node);
        outputs.entry(node).or_default().insert(output);
        coordinates.entry(node).or_default().insert(coordinate);
        let properties = probe["properties"]
            .as_object()
            .ok_or("probe properties are missing")?;
        if id == "wave_000_constant_color" {
            assert_eq!(probe["color_mapping"]["brightness"], 1.25);
            assert_eq!(probe["color_mapping"]["use_color_ramp"], true);
            assert_eq!(probe["color_mapping"]["color_ramp"]["interpolation"], "CONSTANT");
            let element = &probe["color_mapping"]["color_ramp"]["elements"][1];
            assert_json_close(&element["position"], &json!(0.68), "ColorMapping ramp position");
            assert_json_close(
                &element["color"],
                &json!([0.22, 0.72, 0.31, 1.0]),
                "ColorMapping ramp color",
            );
        }
        let vector_active = !(node == "ShaderNodeTexWhiteNoise"
            && properties["noise_dimensions"] == "1D");
        if vector_active {
            let expected_vector = if coordinate == "constant_row9" {
                json!([0.17, 0.62, 0.53])
            } else {
                json!([0.17, 0.31, 0.53])
            };
            assert_probe_input(probe, "Vector", expected_vector)?;
        }
        if node == "ShaderNodeTexBrick" && coordinate != "generated" {
            let vector_y = if coordinate == "constant_row9" { 0.62 } else { 0.31 };
            let row = (vector_y * 4.7_f64 / 0.31).floor() as u32;
            if coordinate == "constant_row9" {
                assert_eq!(row, 9);
                assert_eq!(row % 3, 0, "row 9 exercises Brick offset_frequency=3");
                assert_ne!(row % 4, 0, "row 9 does not exercise squash_frequency=4");
            } else {
                assert_eq!(row, 4);
                assert_eq!(row % 4, 0, "row 4 exercises Brick squash_frequency=4");
                assert_ne!(row % 3, 0, "row 4 does not exercise offset_frequency=3");
            }
        }
        match node {
            "ShaderNodeTexWave" => {
                for (name, expected) in [
                    ("Scale", json!(4.3)),
                    ("Distortion", json!(2.1)),
                    ("Detail", json!(3.0)),
                    ("Detail Scale", json!(2.2)),
                    ("Detail Roughness", json!(0.37)),
                    ("Phase Offset", json!(0.29)),
                ] {
                    assert_probe_input(probe, name, expected)?;
                }
                waves.insert((
                    properties["wave_type"].as_str().ok_or("wave_type is missing")?.to_owned(),
                    properties["wave_profile"].as_str().ok_or("wave_profile is missing")?.to_owned(),
                    if properties["wave_type"] == "BANDS" {
                        properties["bands_direction"].as_str().ok_or("bands_direction is missing")?.to_owned()
                    } else {
                        properties["rings_direction"].as_str().ok_or("rings_direction is missing")?.to_owned()
                    },
                ));
            }
            "ShaderNodeTexMagic" => {
                let depth = properties["turbulence_depth"]
                    .as_i64()
                    .ok_or("Magic turbulence_depth is missing")?;
                magic_depths.insert(depth);
                if depth != 6 {
                    assert_eq!(output, "Fac", "edge-depth probe should use Magic Factor");
                    assert_eq!(coordinate, "constant", "edge-depth probe should use constant coordinates");
                }
                for (name, expected) in [
                    ("Scale", json!(5.1)),
                    ("Distortion", json!(2.3)),
                ] {
                    assert_probe_input(probe, name, expected)?;
                }
            }
            "ShaderNodeTexBrick" => {
                for (name, expected) in [
                    ("offset", json!(0.37)),
                    ("offset_frequency", json!(3)),
                    ("squash", json!(0.73)),
                    ("squash_frequency", json!(4)),
                ] {
                    assert_json_close(&properties[name], &expected, &format!("Brick.{name}"));
                }
                for (name, expected) in [
                    ("Color1", json!([0.72, 0.19, 0.07, 1.0])),
                    ("Color2", json!([0.21, 0.39, 0.68, 1.0])),
                    ("Mortar", json!([0.08, 0.11, 0.16, 1.0])),
                    ("Scale", json!(4.7)),
                    ("Mortar Size", json!(0.12)),
                    ("Mortar Smooth", json!(0.08)),
                    ("Bias", json!(0.27)),
                    ("Brick Width", json!(0.43)),
                    ("Row Height", json!(0.31)),
                ] {
                    assert_probe_input(probe, name, expected)?;
                }
            }
            "ShaderNodeTexWhiteNoise" => {
                white_noise_dimensions.insert(
                    properties["noise_dimensions"].as_str().ok_or("noise_dimensions is missing")?.to_owned(),
                );
                assert_probe_input(probe, "W", json!(0.73))?;
            }
            "ShaderNodeTexGabor" => {
                let dimension = properties["gabor_type"]
                    .as_str()
                    .ok_or("gabor_type is missing")?;
                gabor_dimensions.insert(dimension.to_owned());
                let edge_case = coordinate == "constant_edge";
                if edge_case {
                    assert_eq!(dimension, "3D");
                    assert_eq!(output, "Value");
                }
                for (name, expected) in [
                    ("Scale", json!(3.7)),
                    ("Frequency", json!(2.4)),
                    ("Anisotropy", if edge_case { json!(0.5) } else { json!(0.63) }),
                ] {
                    assert_probe_input(probe, name, expected)?;
                }
                let (orientation, expected) = match (dimension, edge_case) {
                    ("2D", _) => ("Orientation 2D", json!(0.73)),
                    ("3D", true) => ("Orientation 3D", json!([-1.0, 0.0, 0.0])),
                    ("3D", false) => ("Orientation 3D", json!([0.36, 0.48, 0.8])),
                    _ => return Err(format!("unexpected Gabor dimension `{dimension}`").into()),
                };
                assert_probe_input_identifier(probe, orientation, expected)?;
            }
            _ => return Err(format!("unexpected procedural texture node {node}").into()),
        }
    }
    assert_eq!(node_types, expected_nodes);
    assert_eq!(outputs, expected_outputs);
    for node in expected_nodes {
        let expected_sources = if node == "ShaderNodeTexBrick" {
            BTreeSet::from(["constant", "constant_row9", "generated"])
        } else if node == "ShaderNodeTexGabor" {
            BTreeSet::from(["constant", "constant_edge", "generated"])
        } else {
            BTreeSet::from(["constant", "generated"])
        };
        assert_eq!(coordinates[node], expected_sources);
    }
    assert_eq!(
        waves,
        [
            ("BANDS".to_owned(), "SIN".to_owned(), "X".to_owned()),
            ("BANDS".to_owned(), "SIN".to_owned(), "Y".to_owned()),
            ("BANDS".to_owned(), "SIN".to_owned(), "Z".to_owned()),
            ("BANDS".to_owned(), "SIN".to_owned(), "DIAGONAL".to_owned()),
            ("BANDS".to_owned(), "SAW".to_owned(), "X".to_owned()),
            ("BANDS".to_owned(), "SAW".to_owned(), "Y".to_owned()),
            ("BANDS".to_owned(), "SAW".to_owned(), "Z".to_owned()),
            ("BANDS".to_owned(), "SAW".to_owned(), "DIAGONAL".to_owned()),
            ("BANDS".to_owned(), "TRI".to_owned(), "X".to_owned()),
            ("BANDS".to_owned(), "TRI".to_owned(), "Y".to_owned()),
            ("BANDS".to_owned(), "TRI".to_owned(), "Z".to_owned()),
            ("BANDS".to_owned(), "TRI".to_owned(), "DIAGONAL".to_owned()),
            ("RINGS".to_owned(), "SIN".to_owned(), "X".to_owned()),
            ("RINGS".to_owned(), "SIN".to_owned(), "Y".to_owned()),
            ("RINGS".to_owned(), "SIN".to_owned(), "Z".to_owned()),
            ("RINGS".to_owned(), "SIN".to_owned(), "SPHERICAL".to_owned()),
            ("RINGS".to_owned(), "SAW".to_owned(), "X".to_owned()),
            ("RINGS".to_owned(), "SAW".to_owned(), "Y".to_owned()),
            ("RINGS".to_owned(), "SAW".to_owned(), "Z".to_owned()),
            ("RINGS".to_owned(), "SAW".to_owned(), "SPHERICAL".to_owned()),
            ("RINGS".to_owned(), "TRI".to_owned(), "X".to_owned()),
            ("RINGS".to_owned(), "TRI".to_owned(), "Y".to_owned()),
            ("RINGS".to_owned(), "TRI".to_owned(), "Z".to_owned()),
            ("RINGS".to_owned(), "TRI".to_owned(), "SPHERICAL".to_owned()),
        ]
        .into_iter()
        .collect()
    );
    assert_eq!(
        white_noise_dimensions,
        BTreeSet::from(["1D".to_owned(), "2D".to_owned(), "3D".to_owned(), "4D".to_owned()])
    );
    assert_eq!(
        gabor_dimensions,
        BTreeSet::from(["2D".to_owned(), "3D".to_owned()])
    );
    assert_eq!(magic_depths, BTreeSet::from([0_i64, 6, 10]));
    Ok(())
}

fn assert_probe_input(probe: &Value, name: &str, expected: Value) -> TestResult {
    let inputs = probe["inputs"].as_array().ok_or("probe inputs are missing")?;
    let input = inputs
        .iter()
        .find(|input| input["name"] == name)
        .ok_or_else(|| format!("probe input {name} is missing"))?;
    assert_json_close(
        &input["default"],
        &expected,
        &format!("{}.inputs.{name}", probe["id"]),
    );
    Ok(())
}

fn assert_probe_input_identifier(probe: &Value, identifier: &str, expected: Value) -> TestResult {
    let inputs = probe["inputs"].as_array().ok_or("probe inputs are missing")?;
    let input = inputs
        .iter()
        .find(|input| input["identifier"] == identifier)
        .ok_or_else(|| format!("probe input identifier {identifier} is missing"))?;
    assert_json_close(
        &input["default"],
        &expected,
        &format!("{}.inputs.{identifier}", probe["id"]),
    );
    Ok(())
}

fn assert_imported_probes(scene: &Value, oracle: &Value) -> TestResult {
    let probes = oracle["probes"].as_array().ok_or("oracle probes are missing")?;
    for probe in probes {
        let name = probe["node_name"].as_str().ok_or("probe node name is missing")?;
        let (group, node_id, node) = find_node(scene, name)?;
        assert_eq!(node["type"], probe["node_type"], "{name} node type");
        for (key, expected) in probe["properties"].as_object().ok_or("probe properties are missing")? {
            assert_json_close(&node["properties"][key], expected, &format!("{name}.{key}"));
        }
        for key in ["texture_mapping", "color_mapping"] {
            if !probe[key].is_null() {
                assert_json_close(
                    &node["properties"][key],
                    &probe[key],
                    &format!("{name}.{key}"),
                );
            }
        }
        for input in probe["inputs"].as_array().ok_or("probe inputs are missing")? {
            let identifier = input["identifier"]
                .as_str()
                .ok_or("input identifier is missing")?;
            let expected = &input["default"];
            if expected.is_null() {
                continue;
            }
            assert_json_close(
                &node["inputs"][identifier],
                expected,
                &format!("{name}.inputs.{identifier}"),
            );
        }
        let links = group["links"].as_array().ok_or("shader graph links are missing")?;
        let vector_active = !(node["type"] == "ShaderNodeTexWhiteNoise"
            && probe["properties"]["noise_dimensions"] == "1D");
        let vector_linked = links.iter().any(|link| {
            link["to_node"] == node_id && link["to_socket"] == "Vector"
        });
        if !vector_active {
            assert!(!vector_linked, "{name} inactive Vector unexpectedly has an incoming link");
        } else if probe["coordinate_source"] == "generated" {
            assert!(vector_linked, "{name} does not consume Generated coordinates through Vector");
            assert!(
                links.iter().any(|link| {
                    link["from_socket"] == "Generated"
                        && link["to_node"] == node_id
                        && link["to_socket"] == "Vector"
                }),
                "{name} Vector is not linked from Texture Coordinate.Generated"
            );
        } else {
            let vector_link = links
                .iter()
                .find(|link| link["to_node"] == node_id && link["to_socket"] == "Vector")
                .ok_or_else(|| format!("{name} constant Vector has no linked source"))?;
            let source_id = vector_link["from_node"]
                .as_str()
                .ok_or("constant vector source node ID is missing")?;
            let source = &group["nodes"][source_id];
            assert_eq!(source["type"], "ShaderNodeCombineXYZ", "{name} constant vector source");
            assert_eq!(vector_link["from_socket"], "Vector", "{name} constant vector source socket");
            let vector_input = probe["inputs"]
                .as_array()
                .into_iter()
                .flatten()
                .find(|input| input["name"] == "Vector")
                .ok_or("probe Vector input is missing")?;
            let vector: [f64; 3] = serde_json::from_value(vector_input["default"].clone())?;
            for (axis, component) in ["X", "Y", "Z"].into_iter().zip(vector) {
                assert_json_close(
                    &source["inputs"][axis],
                    &json!(component),
                    &format!("{name} constant vector {axis}"),
                );
            }
        }
        if let Some(w_input) = probe["inputs"]
            .as_array()
            .into_iter()
            .flatten()
            .find(|input| input["name"] == "W")
        {
            let w_identifier = w_input["identifier"]
                .as_str()
                .ok_or("W input identifier is missing")?;
            let w_linked = links.iter().any(|link| {
                link["to_node"] == node_id && link["to_socket"] == w_identifier
            });
            let w_enabled = w_input["enabled"]
                .as_bool()
                .ok_or("W socket enabled state is missing")?;
            assert_eq!(
                w_linked,
                w_enabled && probe["coordinate_source"] == "generated",
                "{name} W coordinate link does not match the active probe input"
            );
        }
    }
    Ok(())
}

fn assert_imported_gn_field(scene: &Value, oracle: &Value) -> TestResult {
    let document: SceneDoc = serde_json::from_value(scene.clone())?;
    let gn_oracle = &oracle["gn_field"];
    let group_name = gn_oracle["group_name"]
        .as_str()
        .ok_or("GN texture field graph name is missing")?;
    let group = document
        .node_groups
        .values()
        .find(|group| group.name == group_name)
        .ok_or_else(|| format!("imported GN field graph {group_name} is missing"))?;
    let evaluation = graph::evaluate(group, None, &Map::new(), "Geometry")?;
    let expected_attributes = gn_oracle["attributes"]
        .as_array()
        .ok_or("Blender GN texture field oracle attributes are missing")?;
    assert_eq!(expected_attributes.len(), 5, "GN texture node family coverage");
    for (name, node_type, output, data_type) in [
        ("procedural_wave_fac", "ShaderNodeTexWave", "Fac", "FLOAT"),
        ("procedural_magic_fac", "ShaderNodeTexMagic", "Fac", "FLOAT"),
        ("procedural_brick_color", "ShaderNodeTexBrick", "Color", "FLOAT_COLOR"),
        ("procedural_white_noise_color", "ShaderNodeTexWhiteNoise", "Color", "FLOAT_COLOR"),
        ("procedural_gabor_phase", "ShaderNodeTexGabor", "Phase", "FLOAT"),
    ] {
        let attribute = expected_attributes
            .iter()
            .find(|attribute| attribute["name"] == name)
            .ok_or_else(|| format!("GN oracle is missing {node_type}.{output}"))?;
        assert_eq!(attribute["node_type"], node_type);
        assert_eq!(attribute["output"], output);
        assert_eq!(attribute["data_type"], data_type);
    }
    for expected in expected_attributes {
        let name = expected["name"].as_str().ok_or("GN attribute name is missing")?;
        let values = evaluation.mesh.attributes[name]["data"]
            .as_array()
            .ok_or_else(|| format!("Potter GN attribute {name} values are missing"))?;
        let expected_values = expected["values"]
            .as_array()
            .ok_or_else(|| format!("Blender GN attribute {name} values are missing"))?;
        assert_eq!(values.len(), expected_values.len(), "GN {name} point count");
        for (index, (actual, expected)) in values.iter().zip(expected_values).enumerate() {
            assert_json_close(actual, expected, &format!("Geometry Nodes {name} at point {index}"));
        }
    }
    Ok(())
}

fn find_node<'a>(scene: &'a Value, name: &str) -> TestResult<(&'a Value, &'a str, &'a Value)> {
    for group in scene["node_groups"].as_object().into_iter().flatten().map(|(_, value)| value) {
        if let Some(nodes) = group["nodes"].as_object() {
            for (id, node) in nodes {
                if node["name"].as_str() == Some(name) {
                    return Ok((group, id, node));
                }
            }
        }
    }
    Err(format!("imported procedural texture node {name} is missing").into())
}

fn assert_json_close(actual: &Value, expected: &Value, label: &str) {
    if let (Some(actual), Some(expected)) = (actual.as_f64(), expected.as_f64()) {
        assert_close(actual, expected, 1.0e-5, label);
    } else if let (Some(actual), Some(expected)) = (actual.as_array(), expected.as_array()) {
        assert_eq!(actual.len(), expected.len(), "{label} length");
        for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            assert_json_close(actual, expected, &format!("{label}[{index}]"));
        }
    } else {
        assert_eq!(actual, expected, "{label}");
    }
}

fn render_project(
    project: &Path,
    root: &Path,
    name: &str,
    engine: &str,
    resolution: [u32; 2],
) -> TestResult<ImageData> {
    let scene = read_scene(project)?;
    let active = scene["active_scene"]
        .as_str()
        .ok_or("active scene is missing")?
        .to_owned();
    let updated = apply_operations(
        project,
        root,
        json!([{
            "op": "render.update",
            "target": {"id": active},
            "set": {
                "resolution_x": resolution[0],
                "resolution_y": resolution[1],
                "samples": 16,
                "seed": 0
            }
        }]),
    )?;
    assert_eq!(updated["ok"], true, "{updated}");

    let frames = root.join(name);
    let mut render = pot();
    render
        .arg("render")
        .arg(project)
        .args(["--engine", engine, "--format", "exr", "--out"])
        .arg(&frames)
        .arg("--json");
    checked(&mut render, name)?;

    let image_path = frames.join("frame_0001.exr");
    let loaded = apply_operations(
        project,
        root,
        json!([{
            "op": "image.load",
            "id": format!("{name}_image"),
            "path": image_path.to_string_lossy()
        }]),
    )?;
    assert_eq!(loaded["ok"], true, "{loaded}");
    let scene = read_scene(project)?;
    let image_id = format!("{name}_image");
    let image: Image = serde_json::from_value(scene["images"][&image_id].clone())?;
    let image = load_image_data(&image, project, ImageInterpolation::Closest)?;
    assert_eq!([image.width, image.height], resolution, "Potter render resolution");
    Ok(image)
}

fn compare_probe_pixels(image: &ImageData, oracle: &Value, engine: &str) -> TestResult {
    let probes = oracle["probes"].as_array().ok_or("oracle probes are missing")?;
    let expected_pixels = oracle["renders"][engine]["pixels"]
        .as_array()
        .ok_or("oracle render pixels are missing")?;
    let columns = oracle["columns"].as_u64().ok_or("oracle columns are missing")? as f64;
    let rows = oracle["rows"].as_u64().ok_or("oracle rows are missing")? as f64;
    assert_eq!(probes.len(), expected_pixels.len(), "{engine} oracle probe count");
    for (probe, expected) in probes.iter().zip(expected_pixels) {
        let id = probe["id"].as_str().ok_or("probe id is missing")?;
        let uv = [
            (probe["column"].as_u64().ok_or("probe column is missing")? as f64 + 0.5) / columns,
            (probe["row"].as_u64().ok_or("probe row is missing")? as f64 + 0.5) / rows,
        ];
        let expected: [f64; 4] = serde_json::from_value(expected.clone())?;
        let actual = sample(image, uv, false);
        for channel in 0..4 {
            assert_close(
                actual[channel],
                expected[channel],
                PROBE_PIXEL_TOLERANCE,
                &format!("{engine} {id} channel {channel}"),
            );
        }
    }
    Ok(())
}

fn compare_world_pixels(image: &ImageData, oracle: &Value, engine: &str) -> TestResult {
    let render = &oracle["world_render"]["renders"][engine];
    let width = render["resolution"][0]
        .as_u64()
        .ok_or("world oracle width is missing")? as f64;
    let height = render["resolution"][1]
        .as_u64()
        .ok_or("world oracle height is missing")? as f64;
    let probes = render["probes"]
        .as_array()
        .ok_or("world oracle probes are missing")?;
    let mut first = None;
    let mut varied = false;
    for probe in probes {
        let [x, y]: [u32; 2] = serde_json::from_value(probe["pixel"].clone())?;
        let uv = [
            (f64::from(x) + 0.5) / width,
            (f64::from(y) + 0.5) / height,
        ];
        let expected: [f64; 4] = serde_json::from_value(probe["rgba"].clone())?;
        let actual = sample(image, uv, false);
        for channel in 0..4 {
            assert_close(
                actual[channel],
                expected[channel],
                PROBE_PIXEL_TOLERANCE,
                &format!("{engine} procedural world pixel ({x}, {y}) channel {channel}"),
            );
        }
        if let Some(first) = first {
            varied |= (0..3).any(|channel| (expected[channel] - first[channel]).abs() > 1.0e-3);
        } else {
            first = Some(expected);
        }
    }
    assert!(varied, "{engine} world oracle probes do not distinguish procedural coordinates");
    Ok(())
}
