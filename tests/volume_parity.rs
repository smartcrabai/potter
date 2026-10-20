use std::{cmp::Ordering, collections::HashMap, fs};

use glam::DVec3;
use potter_core::{
    error::ErrorCode,
    eval::{EvaluationContext, Snapshot},
    geom::{Aabb, BoxParams, Mesh, VolumeData, VolumeFileSource, VolumeGrid, VolumeSource, volume},
    hash,
    image::encode_pixels,
    model::{
        DataBlock, Id, Image, ImageAlphaMode, ImageColorspace, ImageSource, Modifier,
        ModifierRuntime, Node, SceneDoc, Transform,
    },
};
use serde::Deserialize;
use serde_json::{Map, json};
use tempfile::tempdir;

const F32_POSITION_TOLERANCE_ULPS: f64 = 1.0;

#[derive(Deserialize)]
struct OracleFile {
    blender_version: String,
    source: OracleSource,
    cases: Vec<OracleCase>,
}

#[derive(Deserialize)]
struct OracleSource {
    amount_diagonal: f64,
    operand_vertices: Vec<[f64; 3]>,
    operand_faces: Vec<Vec<usize>>,
}

#[derive(Deserialize)]
struct OracleCase {
    resolution_mode: String,
    voxel_size: f64,
    voxel_amount: f64,
    interior_band_width: f64,
    density: f64,
    adaptivity: f64,
    grid_names: Vec<String>,
    volume_to_mesh: Vec<OracleMesh>,
}

#[derive(Deserialize)]
struct OracleMesh {
    threshold: f64,
    vertex_count: usize,
    edge_count: usize,
    polygon_count: usize,
    corner_count: usize,
    bounds_min: [f64; 3],
    bounds_max: [f64; 3],
    vertices: Vec<[f64; 3]>,
    polygons: Vec<Vec<usize>>,
    smooth_flags: Vec<bool>,
    #[serde(default)]
    grid_name: Option<String>,
    #[serde(default)]
    adaptivity: Option<f64>,
}

#[derive(Deserialize)]
struct DisplaceOracle {
    blender_version: String,
    image_width: u32,
    image_height: u32,
    image_pixels: Vec<[f64; 4]>,
    cases: Vec<DisplaceCase>,
}

#[derive(Deserialize)]
struct DisplaceCase {
    texture_map_mode: String,
    volume_location: [f64; 3],
    source_location: [f64; 3],
    texture_map_object_location: Option<[f64; 3]>,
    image_colorspace: String,
    image_extension: String,
    image_interpolation: bool,
    strength: f64,
    texture_sample_radius: f64,
    texture_mid_level: [f64; 3],
    voxel_size: f64,
    interior_band_width: f64,
    grid_name: String,
    #[serde(flatten)]
    mesh: OracleMesh,
}

#[derive(Deserialize)]
struct MultiGridVdbOracle {
    blender_version: String,
    vdb_sha256: String,
    voxel_size: f64,
    source_grids: Vec<VdbGridOracle>,
    evaluated_grids: Vec<BlenderGridOracle>,
    texture: OracleImage,
    volume_displace: OracleDisplaceSettings,
    velocity_probe: VectorProbe,
    volume_to_mesh: OracleMesh,
    precision_vdb_sha256: String,
    precision_source_grid: VdbGridOracle,
    precision_volume_to_mesh: OracleMesh,
}

#[derive(Deserialize)]
struct VdbGridOracle {
    name: String,
    value_type: String,
    background: Option<[f64; 3]>,
    grid_class: Option<String>,
    stored_class: Option<String>,
}

#[derive(Deserialize)]
struct BlenderGridOracle {
    name: String,
    data_type: String,
    channels: usize,
}

#[derive(Deserialize)]
struct OracleImage {
    width: u32,
    height: u32,
    pixels: Vec<[f64; 4]>,
}

#[derive(Deserialize)]
struct OracleDisplaceSettings {
    texture_map_mode: String,
    strength: f64,
    texture_sample_radius: f64,
    texture_mid_level: [f64; 3],
}

#[derive(Deserialize)]
struct VectorProbe {
    index: [i32; 3],
    texture_value: [f64; 3],
    displacement_index: [f64; 3],
    expected_sample: [f64; 3],
}

fn fixture(source: &str) -> OracleFile {
    serde_json::from_str(source).expect("valid Blender-generated volume fixture")
}

fn as_dvec3(point: [f64; 3]) -> DVec3 {
    DVec3::new(point[0], point[1], point[2])
}

fn canonical_actual(mesh: &Mesh) -> (Vec<[f64; 3]>, Vec<Vec<usize>>) {
    let mut vertices = mesh
        .vertices
        .iter()
        .map(|vertex| (vertex.id, vertex.co.to_array()))
        .collect::<Vec<_>>();
    vertices.sort_by(|first, second| compare_positions(first.1, second.1));
    let mut vertex_indices = HashMap::with_capacity(vertices.len());
    let positions = vertices
        .iter()
        .enumerate()
        .map(|(index, (id, position))| {
            vertex_indices.insert(*id, index);
            *position
        })
        .collect::<Vec<_>>();
    let mut polygons = mesh
        .faces
        .iter()
        .map(|face| {
            let vertices = face
                .vertices
                .iter()
                .map(|vertex| *vertex_indices.get(vertex).expect("valid mesh face vertex"))
                .collect::<Vec<_>>();
            canonical_rotation(vertices)
        })
        .collect::<Vec<_>>();
    polygons.sort();
    (positions, polygons)
}

fn compare_positions(first: [f64; 3], second: [f64; 3]) -> Ordering {
    first[0]
        .total_cmp(&second[0])
        .then_with(|| first[1].total_cmp(&second[1]))
        .then_with(|| first[2].total_cmp(&second[2]))
}

fn canonical_rotation(vertices: Vec<usize>) -> Vec<usize> {
    (0..vertices.len())
        .map(|offset| {
            vertices[offset..]
                .iter()
                .chain(&vertices[..offset])
                .copied()
                .collect::<Vec<_>>()
        })
        .min()
        .expect("polygon has at least three vertices")
}

fn assert_mesh_matches(oracle: &OracleMesh, actual: &Mesh, context: &str) {
    let (vertices, polygons) = canonical_actual(actual);
    assert_eq!(
        vertices.len(),
        oracle.vertex_count,
        "{context}: vertex count"
    );
    assert_eq!(
        actual.edges.len(),
        oracle.edge_count,
        "{context}: edge count"
    );
    assert_eq!(
        polygons.len(),
        oracle.polygon_count,
        "{context}: polygon count"
    );
    assert_eq!(
        actual
            .faces
            .iter()
            .map(|face| face.vertices.len())
            .sum::<usize>(),
        oracle.corner_count,
        "{context}: corner count"
    );
    for (index, (actual, expected)) in vertices.iter().zip(&oracle.vertices).enumerate() {
        for axis in 0..3 {
            let tolerance = F32_POSITION_TOLERANCE_ULPS
                * f64::from(f32::EPSILON)
                * expected[axis].abs().max(1.0);
            assert!(
                (actual[axis] - expected[axis]).abs() <= tolerance,
                "{context}: vertex {index} axis {axis}: actual {}, Blender {}, tolerance {tolerance}",
                actual[axis],
                expected[axis]
            );
        }
    }
    assert_eq!(
        polygons, oracle.polygons,
        "{context}: full corner association"
    );
    for axis in 0..3 {
        let (actual_min, actual_max) = actual
            .bounds()
            .map(|bounds| (bounds.min[axis], bounds.max[axis]))
            .expect("nonempty volume mesh bounds");
        assert!(
            (actual_min - oracle.bounds_min[axis]).abs()
                <= F32_POSITION_TOLERANCE_ULPS * f64::from(f32::EPSILON)
        );
        assert!(
            (actual_max - oracle.bounds_max[axis]).abs()
                <= F32_POSITION_TOLERANCE_ULPS * f64::from(f32::EPSILON)
        );
    }
}

fn check_fixture(file: OracleFile, shape: &str) {
    assert_eq!(file.blender_version, "5.2.2 LTS");
    let positions = file
        .source
        .operand_vertices
        .into_iter()
        .map(as_dvec3)
        .collect();
    let mesh = Mesh::from_positions_and_faces(positions, file.source.operand_faces)
        .expect("valid closed Blender fixture mesh");
    assert_eq!(
        file.cases.len(),
        32,
        "{shape}: resolution/band/density/adaptivity matrix"
    );
    for case in file.cases {
        assert_eq!(
            case.grid_names,
            vec!["density".to_owned()],
            "{shape}: generated grid name"
        );
        let voxel_size = match case.resolution_mode.as_str() {
            "VOXEL_SIZE" => case.voxel_size,
            "VOXEL_AMOUNT" => {
                let denominator = (case.voxel_amount as f32).max(1.0);
                f64::from((file.source.amount_diagonal as f32 / denominator) as f32)
            }
            mode => panic!("{shape}: unexpected resolution mode {mode}"),
        };
        let volume =
            volume::mesh_to_fog_volume(&mesh, voxel_size, case.interior_band_width, case.density)
                .expect("Potter mesh-to-volume conversion");
        for oracle in &case.volume_to_mesh {
            assert_eq!(oracle.smooth_flags.len(), oracle.polygon_count);
            assert!(oracle.smooth_flags.iter().all(|smooth| *smooth));
            let context = format!(
                "{shape} {} band={} density={} adaptivity={} threshold={}",
                case.resolution_mode,
                case.interior_band_width,
                case.density,
                case.adaptivity,
                oracle.threshold
            );
            let actual = volume::volume_to_mesh_named_adaptive(
                &volume,
                oracle.threshold,
                "density",
                case.adaptivity,
            )
            .unwrap_or_else(|error| panic!("{context}: {error}"));
            assert_mesh_matches(oracle, &actual, &context);
        }
    }
}

#[test]
fn blender_5_2_2_volume_modifier_meshes_match_fixtures() {
    check_fixture(
        fixture(include_str!("fixtures/pr18o_volume/asymmetric_box.json")),
        "asymmetric_box",
    );
    check_fixture(
        fixture(include_str!(
            "fixtures/pr18o_volume/rotated_nonuniform.json"
        )),
        "rotated_nonuniform",
    );
    check_fixture(
        fixture(include_str!("fixtures/pr18o_volume/tiny_voxels.json")),
        "tiny_voxels",
    );
}

#[test]
fn invalid_voxel_size_produces_no_mesh_to_volume_grid() {
    let mesh = Mesh::box_mesh(Default::default()).expect("valid cube");
    let result = volume::mesh_to_fog_volume(&mesh, 1.0e-6, 0.2, 1.0)
        .expect("invalid modifier voxel size creates no grid");
    assert!(result.grids.is_empty());
}

fn volume_modifier_doc(resolution_mode: &str, size: f64, amount: f64) -> (SceneDoc, Id, Id) {
    let source_node_id = Id::new("mesh_source").expect("valid source node ID");
    let volume_node_id = Id::new("volume_owner").expect("valid volume node ID");
    let source_data_id = Id::new("mesh_data").expect("valid mesh data ID");
    let volume_data_id = Id::new("volume_data").expect("valid volume data ID");
    let mut doc = SceneDoc::new("00000000-0000-4000-8000-000000000001".to_owned());
    let source_mesh = Mesh::box_mesh(BoxParams {
        size: DVec3::splat(2.0),
    })
    .expect("valid operand cube");
    doc.data_blocks.insert(
        source_data_id.clone(),
        DataBlock {
            data_type: "mesh".to_owned(),
            mesh: Some(source_mesh),
            ..DataBlock::default()
        },
    );
    doc.data_blocks.insert(
        volume_data_id.clone(),
        DataBlock {
            data_type: "volume".to_owned(),
            volume: Some(VolumeData {
                grids: vec![VolumeGrid {
                    dims: [2, 2, 2],
                    voxel_size: 1.0,
                    origin: DVec3::ZERO,
                    values: Some(vec![1.0; 8]),
                    ..VolumeGrid::default()
                }],
                ..VolumeData::default()
            }),
            ..DataBlock::default()
        },
    );
    let source_node = Node {
        name: "operand".to_owned(),
        kind: "mesh".to_owned(),
        data: Some(source_data_id),
        ..Node::default()
    };
    let mut params = Map::new();
    params.insert("object".to_owned(), json!(source_node_id.as_str()));
    params.insert("resolution_mode".to_owned(), json!(resolution_mode));
    params.insert("voxel_size".to_owned(), json!(size));
    params.insert("voxel_amount".to_owned(), json!(amount));
    params.insert("interior_band_width".to_owned(), json!(0.2));
    params.insert("density".to_owned(), json!(1.0));
    let modifier = Modifier {
        id: Id::new("modifier_mtv").expect("valid modifier ID"),
        modifier_type: "mesh_to_volume".to_owned(),
        name: "Mesh to Volume".to_owned(),
        enabled: true,
        show_viewport: true,
        show_render: true,
        params,
        binding_data: None,
        runtime: ModifierRuntime::default(),
    };
    let volume_node = Node {
        name: "volume".to_owned(),
        kind: "volume".to_owned(),
        data: Some(volume_data_id.clone()),
        modifiers: vec![modifier],
        ..Node::default()
    };
    doc.nodes.insert(source_node_id.clone(), source_node);
    doc.nodes.insert(volume_node_id.clone(), volume_node);
    let root_collection_id = Id::new("collection_root").expect("valid root collection ID");
    doc.collections
        .get_mut(&root_collection_id)
        .expect("scene root collection")
        .objects
        .extend([source_node_id.clone(), volume_node_id.clone()]);
    (doc, volume_node_id, volume_data_id)
}

#[test]
fn invalid_mesh_to_volume_settings_leave_existing_volume_unchanged() {
    for (resolution_mode, voxel_size, voxel_amount) in
        [("VOXEL_SIZE", 1.0e-6, 32.0), ("VOXEL_AMOUNT", 0.1, 0.5)]
    {
        let (doc, volume_node_id, volume_data_id) =
            volume_modifier_doc(resolution_mode, voxel_size, voxel_amount);
        let snapshot = Snapshot::evaluate(&doc, &EvaluationContext::default())
            .expect("inactive mesh-to-volume settings leave the volume evaluable");
        assert!(
            !snapshot.volume_data.contains_key(&volume_data_id),
            "{resolution_mode} should not replace the input volume"
        );
        assert_eq!(
            snapshot
                .nodes
                .get(&volume_node_id)
                .expect("evaluated volume node")
                .bounds,
            Some(Aabb {
                min: DVec3::ZERO,
                max: DVec3::ONE,
            }),
            "{resolution_mode} should preserve the input volume bounds"
        );
    }
}

#[test]
fn volume_nodes_reject_legacy_volume_to_mesh_modifiers() {
    let (mut doc, volume_node_id, _) = volume_modifier_doc("VOXEL_SIZE", 0.25, 32.0);
    doc.nodes
        .get_mut(&volume_node_id)
        .expect("volume node")
        .modifiers[0]
        .modifier_type = "volume_to_mesh".to_owned();
    let error = Snapshot::evaluate(&doc, &EvaluationContext::default())
        .expect_err("Volume to Mesh is not valid on a Volume object");
    assert_eq!(error.code, ErrorCode::InvalidOperation);
    assert_eq!(
        error.message,
        "volume_to_mesh modifier can only be used on mesh objects"
    );
}

#[test]
fn mesh_nodes_reject_volume_only_modifiers() {
    let (mut doc, _, _) = volume_modifier_doc("VOXEL_SIZE", 0.25, 32.0);
    let source_node_id = Id::new("mesh_source").expect("valid source node ID");
    doc.nodes
        .get_mut(&source_node_id)
        .expect("mesh source node")
        .modifiers
        .push(Modifier {
            id: Id::new("modifier_vd").expect("valid modifier ID"),
            modifier_type: "volume_displace".to_owned(),
            name: "Volume Displace".to_owned(),
            enabled: true,
            show_viewport: true,
            show_render: true,
            params: Map::new(),
            binding_data: None,
            runtime: ModifierRuntime::default(),
        });
    let error = Snapshot::evaluate(&doc, &EvaluationContext::default())
        .expect_err("Volume Displace is not valid on a mesh object");
    assert_eq!(error.code, ErrorCode::InvalidOperation);
    assert_eq!(
        error.message,
        "volume_displace modifier can only be used on volume objects"
    );
}

fn image_displace_scene(case: &DisplaceCase, image_id: &Id, image_blob: &str) -> (SceneDoc, Id) {
    let source_node_id = Id::new("mesh_source").expect("valid source node ID");
    let volume_node_id = Id::new("volume_owner").expect("valid volume node ID");
    let carrier_node_id = Id::new("mesh_result").expect("valid carrier node ID");
    let source_data_id = Id::new("mesh_data").expect("valid mesh data ID");
    let volume_data_id = Id::new("volume_data").expect("valid volume data ID");
    let carrier_data_id = Id::new("carrier_data").expect("valid carrier data ID");
    let mut doc = SceneDoc::new("00000000-0000-4000-8000-000000000001".to_owned());
    doc.images.insert(
        image_id.clone(),
        Image {
            name: "gradient".to_owned(),
            source: ImageSource::Generated,
            colorspace: ImageColorspace::Linear,
            width: 16,
            height: 16,
            tiles: Vec::new(),
            blob: Some(image_blob.to_owned()),
            source_path: None,
            source_hash: None,
            alpha_mode: ImageAlphaMode::Straight,
        },
    );
    doc.data_blocks.insert(
        source_data_id.clone(),
        DataBlock {
            data_type: "mesh".to_owned(),
            mesh: Some(
                Mesh::box_mesh(BoxParams {
                    size: DVec3::splat(2.0),
                })
                .expect("valid source cube"),
            ),
            ..DataBlock::default()
        },
    );
    doc.data_blocks.insert(
        volume_data_id.clone(),
        DataBlock {
            data_type: "volume".to_owned(),
            volume: Some(VolumeData {
                grids: vec![VolumeGrid {
                    values: Some(vec![0.0]),
                    ..VolumeGrid::default()
                }],
                ..VolumeData::default()
            }),
            ..DataBlock::default()
        },
    );
    doc.data_blocks.insert(
        carrier_data_id.clone(),
        DataBlock {
            data_type: "mesh".to_owned(),
            mesh: Some(Mesh::default()),
            ..DataBlock::default()
        },
    );
    let source_node = Node {
        name: "source".to_owned(),
        kind: "mesh".to_owned(),
        data: Some(source_data_id),
        transform: Transform {
            translation: case.source_location,
            ..Transform::default()
        },
        ..Node::default()
    };
    let mut mesh_to_volume_params = Map::new();
    mesh_to_volume_params.insert("object".to_owned(), json!(source_node_id.as_str()));
    mesh_to_volume_params.insert("resolution_mode".to_owned(), json!("VOXEL_SIZE"));
    mesh_to_volume_params.insert("voxel_size".to_owned(), json!(case.voxel_size));
    mesh_to_volume_params.insert(
        "interior_band_width".to_owned(),
        json!(case.interior_band_width),
    );
    let mut displace_params = Map::new();
    displace_params.insert("texture".to_owned(), json!(image_id.as_str()));
    displace_params.insert("strength".to_owned(), json!(case.strength));
    displace_params.insert("texture_map_mode".to_owned(), json!(case.texture_map_mode));
    displace_params.insert(
        "texture_sample_radius".to_owned(),
        json!(case.texture_sample_radius),
    );
    displace_params.insert(
        "texture_mid_level".to_owned(),
        json!(case.texture_mid_level),
    );
    let mut volume_modifiers = vec![
        Modifier {
            id: Id::new("modifier_mtv").expect("valid modifier ID"),
            modifier_type: "mesh_to_volume".to_owned(),
            name: "Mesh to Volume".to_owned(),
            enabled: true,
            show_viewport: true,
            show_render: true,
            params: mesh_to_volume_params,
            binding_data: None,
            runtime: ModifierRuntime::default(),
        },
        Modifier {
            id: Id::new("modifier_vd").expect("valid modifier ID"),
            modifier_type: "volume_displace".to_owned(),
            name: "Volume Displace".to_owned(),
            enabled: true,
            show_viewport: true,
            show_render: true,
            params: displace_params,
            binding_data: None,
            runtime: ModifierRuntime::default(),
        },
    ];
    let map_node_id = case.texture_map_object_location.as_ref().map(|_| {
        let id = Id::new("texture_map_object").expect("valid texture map object ID");
        volume_modifiers[1]
            .params
            .insert("texture_map_object".to_owned(), json!(id.as_str()));
        id
    });
    let volume_node = Node {
        name: "volume".to_owned(),
        kind: "volume".to_owned(),
        data: Some(volume_data_id),
        transform: Transform {
            translation: case.volume_location,
            ..Transform::default()
        },
        modifiers: volume_modifiers,
        ..Node::default()
    };
    let mut carrier_params = Map::new();
    carrier_params.insert("object".to_owned(), json!(volume_node_id.as_str()));
    carrier_params.insert("grid_name".to_owned(), json!(case.grid_name));
    carrier_params.insert("threshold".to_owned(), json!(case.mesh.threshold));
    carrier_params.insert("adaptivity".to_owned(), json!(0.0));
    let carrier_node = Node {
        name: "carrier".to_owned(),
        kind: "mesh".to_owned(),
        data: Some(carrier_data_id),
        modifiers: vec![Modifier {
            id: Id::new("modifier_vtm").expect("valid modifier ID"),
            modifier_type: "volume_to_mesh".to_owned(),
            name: "Volume to Mesh".to_owned(),
            enabled: true,
            show_viewport: true,
            show_render: true,
            params: carrier_params,
            binding_data: None,
            runtime: ModifierRuntime::default(),
        }],
        ..Node::default()
    };
    doc.nodes.insert(source_node_id.clone(), source_node);
    doc.nodes.insert(volume_node_id.clone(), volume_node);
    doc.nodes.insert(carrier_node_id.clone(), carrier_node);
    let mut object_ids = vec![source_node_id, volume_node_id, carrier_node_id.clone()];
    if let (Some(map_node_id), Some(location)) = (map_node_id, case.texture_map_object_location) {
        doc.nodes.insert(
            map_node_id.clone(),
            Node {
                name: "map".to_owned(),
                kind: "empty".to_owned(),
                transform: Transform {
                    translation: location,
                    ..Transform::default()
                },
                ..Node::default()
            },
        );
        object_ids.push(map_node_id);
    }
    let root_collection_id = Id::new("collection_root").expect("valid root collection ID");
    doc.collections
        .get_mut(&root_collection_id)
        .expect("scene root collection")
        .objects
        .extend(object_ids);
    (doc, carrier_node_id)
}

#[test]
fn blender_volume_displace_image_mapping_modes_match_full_mesh_fixtures() {
    let oracle: DisplaceOracle = serde_json::from_str(include_str!(
        "fixtures/pr18o_volume/volume_displace_image.json"
    ))
    .expect("valid Blender image displacement fixture");
    assert_eq!(oracle.blender_version, "5.2.2 LTS");
    assert_eq!([oracle.image_width, oracle.image_height], [16, 16]);
    assert_eq!(oracle.cases.len(), 3);
    let pixels = encode_pixels(
        oracle.image_width,
        oracle.image_height,
        &oracle.image_pixels,
    )
    .expect("valid Blender gradient image pixels");
    let image_blob = hash::sha256(&pixels);
    let project = tempdir().expect("temporary image asset project");
    let blob_hex = image_blob.strip_prefix("sha256:").expect("SHA-256 blob ID");
    let blob_path = project
        .path()
        .join("assets")
        .join("sha256")
        .join(blob_hex)
        .join("blob");
    fs::create_dir_all(blob_path.parent().expect("asset parent"))
        .expect("create image asset directory");
    fs::write(&blob_path, pixels).expect("write image asset");
    let image_id = Id::new("image_gradient").expect("valid image ID");
    for case in &oracle.cases {
        assert_eq!(case.image_colorspace, "Linear Rec.709");
        assert_eq!(case.image_extension, "REPEAT");
        assert!(case.image_interpolation);
        let (doc, carrier_id) = image_displace_scene(case, &image_id, &image_blob);
        let snapshot = Snapshot::evaluate_with_cache(
            &doc,
            &EvaluationContext::default(),
            Some(project.path()),
            None,
        )
        .unwrap_or_else(|error| panic!("{} Volume Displace: {error}", case.texture_map_mode));
        let actual = snapshot
            .meshes
            .get(&carrier_id)
            .expect("Volume to Mesh generated carrier geometry");
        assert_mesh_matches(
            &case.mesh,
            actual,
            &format!("Volume Displace {}", case.texture_map_mode),
        );
    }
}

#[test]
fn blender_multigrid_vdb_displacement_preserves_named_grid_types_and_samples() {
    let oracle: MultiGridVdbOracle = serde_json::from_str(include_str!(
        "fixtures/pr18o_volume/volume_multigrid_vdb.json"
    ))
    .expect("valid Blender multi-grid VDB fixture");
    assert_eq!(oracle.blender_version, "5.2.2 LTS");
    assert_eq!(oracle.volume_displace.texture_map_mode, "LOCAL");
    assert_eq!(oracle.volume_to_mesh.polygon_count, 170);
    assert_eq!(
        oracle
            .evaluated_grids
            .iter()
            .map(|grid| grid.name.as_str())
            .collect::<Vec<_>>(),
        ["density", "velocity"]
    );

    let directory = tempfile::tempdir().expect("temporary VDB evaluation resources");
    let vdb_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/pr18o_volume/multigrid.vdb");
    let vdb_bytes = fs::read(&vdb_path).expect("checked-in VDB fixture");
    assert_eq!(hash::sha256(&vdb_bytes), oracle.vdb_sha256);
    let vdb_resource_id = Id::new("multigrid_vdb").expect("valid VDB resource ID");
    let precision_vdb_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/pr18o_volume/precision.vdb");
    let precision_vdb_bytes =
        fs::read(&precision_vdb_path).expect("checked-in precision VDB fixture");
    assert_eq!(
        hash::sha256(&precision_vdb_bytes),
        oracle.precision_vdb_sha256
    );
    let precision_vdb_resource_id =
        Id::new("precision_vdb").expect("valid precision VDB resource ID");
    let image_id = Id::new("gradient_image").expect("valid image ID");
    let image_bytes = encode_pixels(
        oracle.texture.width,
        oracle.texture.height,
        &oracle.texture.pixels,
    )
    .expect("valid gradient image pixels");
    let image_hash = hash::sha256(&image_bytes);
    let image_hex = image_hash
        .strip_prefix("sha256:")
        .expect("SHA-256 image hash");
    let image_path = directory
        .path()
        .join("assets")
        .join("sha256")
        .join(image_hex)
        .join("blob");
    fs::create_dir_all(image_path.parent().expect("image asset directory"))
        .expect("create image asset directory");
    fs::write(&image_path, image_bytes).expect("write gradient image asset");

    let mut doc = SceneDoc::new("00000000-0000-4000-8000-000000000001".to_owned());
    doc.resources.insert(
        vdb_resource_id.clone(),
        json!({
            "uri": vdb_path.to_string_lossy(),
            "hash": oracle.vdb_sha256,
            "kind": "vdb",
            "packed": false
        }),
    );
    doc.resources.insert(
        precision_vdb_resource_id.clone(),
        json!({
            "uri": precision_vdb_path.to_string_lossy(),
            "hash": oracle.precision_vdb_sha256,
            "kind": "vdb",
            "packed": false
        }),
    );
    doc.images.insert(
        image_id.clone(),
        Image {
            name: "multi-grid displacement texture".to_owned(),
            source: ImageSource::Generated,
            colorspace: ImageColorspace::Linear,
            width: oracle.texture.width,
            height: oracle.texture.height,
            tiles: Vec::new(),
            blob: Some(image_hash),
            source_path: None,
            source_hash: None,
            alpha_mode: ImageAlphaMode::Straight,
        },
    );

    let volume_node_id = Id::new("vdb_volume").expect("valid VDB volume node ID");
    let volume_data_id = Id::new("vdb_data").expect("valid VDB data ID");
    let mesh_node_id = Id::new("vdb_surface").expect("valid surface node ID");
    let mesh_data_id = Id::new("vdb_surface_data").expect("valid surface data ID");
    let precision_volume_node_id =
        Id::new("precision_volume").expect("valid precision volume node ID");
    let precision_volume_data_id = Id::new("precision_data").expect("valid precision data ID");
    let precision_mesh_node_id = Id::new("precision_surface").expect("valid precision surface ID");
    let precision_mesh_data_id =
        Id::new("precision_surface_data").expect("valid precision mesh data ID");
    let source = VolumeSource::File(VolumeFileSource {
        format: "vdb".to_owned(),
        content_ref: Some(format!("asset://{}", vdb_resource_id.as_str())),
        grid_names: oracle
            .source_grids
            .iter()
            .map(|grid| grid.name.clone())
            .collect(),
        bounds_min: None,
        bounds_max: None,
    });
    doc.data_blocks.insert(
        volume_data_id.clone(),
        DataBlock {
            data_type: "volume".to_owned(),
            volume: Some(VolumeData {
                source,
                ..VolumeData::default()
            }),
            ..DataBlock::default()
        },
    );
    doc.data_blocks.insert(
        mesh_data_id.clone(),
        DataBlock {
            data_type: "mesh".to_owned(),
            mesh: Some(Mesh::default()),
            ..DataBlock::default()
        },
    );
    doc.data_blocks.insert(
        precision_volume_data_id.clone(),
        DataBlock {
            data_type: "volume".to_owned(),
            volume: Some(VolumeData {
                source: VolumeSource::File(VolumeFileSource {
                    format: "vdb".to_owned(),
                    content_ref: Some(format!("asset://{}", precision_vdb_resource_id.as_str())),
                    grid_names: vec![oracle.precision_source_grid.name.clone()],
                    bounds_min: None,
                    bounds_max: None,
                }),
                ..VolumeData::default()
            }),
            ..DataBlock::default()
        },
    );
    doc.data_blocks.insert(
        precision_mesh_data_id.clone(),
        DataBlock {
            data_type: "mesh".to_owned(),
            mesh: Some(Mesh::default()),
            ..DataBlock::default()
        },
    );

    let mut precision_mesh_params = Map::new();
    precision_mesh_params.insert(
        "object".to_owned(),
        json!(precision_volume_node_id.as_str()),
    );
    precision_mesh_params.insert(
        "grid_name".to_owned(),
        json!(
            oracle
                .precision_volume_to_mesh
                .grid_name
                .as_deref()
                .expect("precision VDB oracle includes grid name")
        ),
    );
    precision_mesh_params.insert(
        "threshold".to_owned(),
        json!(oracle.precision_volume_to_mesh.threshold),
    );
    precision_mesh_params.insert(
        "adaptivity".to_owned(),
        json!(
            oracle
                .precision_volume_to_mesh
                .adaptivity
                .expect("precision VDB oracle includes adaptivity")
        ),
    );
    let mut displace_params = Map::new();
    displace_params.insert("texture".to_owned(), json!(image_id.as_str()));
    displace_params.insert("texture_map_mode".to_owned(), json!("LOCAL"));
    displace_params.insert(
        "strength".to_owned(),
        json!(oracle.volume_displace.strength),
    );
    displace_params.insert(
        "texture_sample_radius".to_owned(),
        json!(oracle.volume_displace.texture_sample_radius),
    );
    displace_params.insert(
        "texture_mid_level".to_owned(),
        json!(oracle.volume_displace.texture_mid_level),
    );
    let mut mesh_params = Map::new();
    mesh_params.insert("object".to_owned(), json!(volume_node_id.as_str()));
    mesh_params.insert(
        "grid_name".to_owned(),
        json!(
            oracle
                .volume_to_mesh
                .grid_name
                .as_deref()
                .expect("VDB oracle includes grid name")
        ),
    );
    mesh_params.insert(
        "threshold".to_owned(),
        json!(oracle.volume_to_mesh.threshold),
    );
    mesh_params.insert(
        "adaptivity".to_owned(),
        json!(
            oracle
                .volume_to_mesh
                .adaptivity
                .expect("VDB oracle includes adaptivity")
        ),
    );
    doc.nodes.insert(
        volume_node_id.clone(),
        Node {
            name: "VDB Volume".to_owned(),
            kind: "volume".to_owned(),
            data: Some(volume_data_id.clone()),
            modifiers: vec![Modifier {
                id: Id::new("vdb_displace").expect("valid displacement modifier ID"),
                modifier_type: "volume_displace".to_owned(),
                name: "Displace both VDB grids".to_owned(),
                enabled: true,
                show_viewport: true,
                show_render: true,
                params: displace_params,
                binding_data: None,
                runtime: ModifierRuntime::default(),
            }],
            ..Node::default()
        },
    );
    doc.nodes.insert(
        mesh_node_id.clone(),
        Node {
            name: "Density surface".to_owned(),
            kind: "mesh".to_owned(),
            data: Some(mesh_data_id),
            modifiers: vec![Modifier {
                id: Id::new("vdb_extract").expect("valid VDB extraction modifier ID"),
                modifier_type: "volume_to_mesh".to_owned(),
                name: "Extract density grid".to_owned(),
                enabled: true,
                show_viewport: true,
                show_render: true,
                params: mesh_params,
                binding_data: None,
                runtime: ModifierRuntime::default(),
            }],
            ..Node::default()
        },
    );
    doc.nodes.insert(
        precision_volume_node_id.clone(),
        Node {
            name: "Double Precision Volume".to_owned(),
            kind: "volume".to_owned(),
            data: Some(precision_volume_data_id.clone()),
            ..Node::default()
        },
    );
    doc.nodes.insert(
        precision_mesh_node_id.clone(),
        Node {
            name: "Double Precision Surface".to_owned(),
            kind: "mesh".to_owned(),
            data: Some(precision_mesh_data_id),
            modifiers: vec![Modifier {
                id: Id::new("precision_extract").expect("valid precision extraction modifier ID"),
                modifier_type: "volume_to_mesh".to_owned(),
                name: "Extract double precision grid".to_owned(),
                enabled: true,
                show_viewport: true,
                show_render: true,
                params: precision_mesh_params,
                binding_data: None,
                runtime: ModifierRuntime::default(),
            }],
            ..Node::default()
        },
    );
    let root_collection_id = Id::new("collection_root").expect("valid root collection ID");
    doc.collections
        .get_mut(&root_collection_id)
        .expect("scene root collection")
        .objects
        .extend([
            volume_node_id,
            mesh_node_id.clone(),
            precision_volume_node_id,
            precision_mesh_node_id.clone(),
        ]);

    let snapshot = Snapshot::evaluate_with_cache(
        &doc,
        &EvaluationContext::default(),
        Some(directory.path()),
        None,
    )
    .expect("multi-grid VDB displacement and named density extraction");
    let actual_mesh = snapshot
        .meshes
        .get(&mesh_node_id)
        .expect("named density grid produced a mesh");
    assert_mesh_matches(
        &oracle.volume_to_mesh,
        actual_mesh,
        "multi-grid VDB density displacement",
    );

    let mut vector_mesh_doc = doc.clone();
    vector_mesh_doc
        .nodes
        .get_mut(&mesh_node_id)
        .expect("surface node")
        .modifiers[0]
        .params
        .insert("grid_name".to_owned(), json!("velocity"));
    let vector_snapshot = Snapshot::evaluate_with_cache(
        &vector_mesh_doc,
        &EvaluationContext::default(),
        Some(directory.path()),
        None,
    )
    .expect("vector VDB grids are valid but not meshable scalar fields");
    let vector_mesh = vector_snapshot
        .meshes
        .get(&mesh_node_id)
        .expect("VOLUME_TO_MESH target node remains evaluated");
    assert!(
        vector_mesh.vertices.is_empty() && vector_mesh.faces.is_empty(),
        "Blender returns an empty mesh for vector grids"
    );

    let precision_mesh = snapshot
        .meshes
        .get(&precision_mesh_node_id)
        .expect("standalone double scalar VOLUME_TO_MESH result");
    assert_mesh_matches(
        &oracle.precision_volume_to_mesh,
        precision_mesh,
        "double-precision VDB VOLUME_TO_MESH",
    );
    let precision_grid = snapshot
        .volume_data
        .get(&precision_volume_data_id)
        .and_then(|volume| volume.decoded_vdb.as_ref())
        .and_then(|vdb| vdb.grids.iter().find(|grid| grid.name == "precision"))
        .expect("double precision VDB grid retained");
    assert!(precision_grid.value_type.contains("double"));

    let displaced = snapshot
        .volume_data
        .get(&volume_data_id)
        .expect("file-backed VDB displacement result");
    assert!(matches!(&displaced.source, VolumeSource::File(_)));
    let decoded = displaced
        .decoded_vdb
        .as_ref()
        .expect("displaced VDB grids remain typed and decoded");
    let actual_names = decoded
        .grids
        .iter()
        .map(|grid| grid.name.as_str())
        .collect::<Vec<_>>();
    let expected_names = oracle
        .evaluated_grids
        .iter()
        .map(|grid| grid.name.as_str())
        .collect::<Vec<_>>();
    assert_eq!(actual_names, expected_names, "VDB grid order and names");
    for blender_grid in &oracle.evaluated_grids {
        let vdb_grid = decoded
            .grids
            .iter()
            .find(|grid| grid.name == blender_grid.name)
            .expect("Blender-evaluated VDB grid remains present");
        match blender_grid.data_type.as_str() {
            "FLOAT" => {
                assert_eq!(blender_grid.channels, 1);
                assert!(vdb_grid.value_type.contains("float"));
                assert!(!vdb_grid.value_type.contains("vec3"));
            }
            "VECTOR_FLOAT" => {
                assert_eq!(blender_grid.channels, 3);
                assert!(vdb_grid.value_type.contains("vec3s"));
            }
            kind => panic!("unexpected Blender VDB grid type {kind}"),
        }
    }
    for expected in &oracle.source_grids {
        let grid = decoded
            .grids
            .iter()
            .find(|grid| grid.name == expected.name)
            .expect("displacement retained every named VDB grid");
        assert_eq!(
            grid.value_type, expected.value_type,
            "{} type",
            expected.name
        );
        if let Some(grid_class) = expected.grid_class.as_deref() {
            assert_eq!(grid_class, "FOG_VOLUME");
            assert_eq!(expected.name, "density");
            if let Some(stored_class) = expected.stored_class.as_deref() {
                assert_eq!(grid.class.as_deref(), Some(stored_class));
            }
        }
    }
    let vector_grid = decoded
        .grids
        .iter()
        .find(|grid| grid.name == "velocity")
        .expect("vector VDB grid retained");
    assert_eq!(oracle.velocity_probe.index, [3, 3, 3]);
    let mut expected_sample = [0.0; 3];
    for axis in 0..3 {
        let texture_value = oracle.velocity_probe.texture_value[axis];
        let texture_reference = [0.9, 0.9, 0.5][axis];
        let tolerance = F32_POSITION_TOLERANCE_ULPS
            * f64::from(f32::EPSILON)
            * texture_reference.abs().max(1.0);
        assert!(
            (texture_value - texture_reference).abs() <= tolerance,
            "texture component {axis}: {texture_value}"
        );
        let displacement = (texture_value - oracle.volume_displace.texture_mid_level[axis])
            * oracle.volume_displace.strength
            / oracle.voxel_size;
        let recorded_displacement = oracle.velocity_probe.displacement_index[axis];
        assert!(
            (displacement - recorded_displacement).abs() <= tolerance,
            "displacement component {axis}: {displacement} vs {recorded_displacement}"
        );
        expected_sample[axis] = (f64::from(oracle.velocity_probe.index[axis]) - displacement) / 7.0;
        assert!(
            (expected_sample[axis] - oracle.velocity_probe.expected_sample[axis]).abs()
                <= tolerance,
            "fixture velocity sample component {axis} is inconsistent"
        );
        let actual = vector_grid
            .sample_index_component(oracle.velocity_probe.index, axis)
            .expect("typed vector sample component");
        assert!(
            (f64::from(actual) - expected_sample[axis]).abs() <= tolerance,
            "velocity component {axis}: got {actual}, expected {}",
            expected_sample[axis]
        );
    }
}
