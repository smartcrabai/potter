use std::{cmp::Ordering, collections::HashMap, error::Error, sync::Arc};

use glam::{DMat4, DVec3};
use potter_core::{
    error::ErrorCode,
    geom::{
        Mesh,
        vdb::VdbVolume,
        volume::{
            VolumeData, VolumeFileSource, VolumeSource, VolumeToMeshResolution,
            sample_density_checked_with, volume_to_mesh_named_adaptive, volume_to_mesh_placed,
        },
    },
};
use serde_json::Value;

const EXPECTED: &str = include_str!("fixtures/openvdb_transform_types/expected.json");
const VALUE_TYPES: &[u8] = include_bytes!("fixtures/openvdb_transform_types/value_types.vdb");
const FRUSTUM: &[u8] = include_bytes!("fixtures/openvdb_transform_types/frustum.vdb");
const DENSITY: &[u8] = include_bytes!("fixtures/openvdb_transform_types/density.vdb");

fn close(actual: f64, expected: f64, tolerance: f64) {
    assert!(
        (actual - expected).abs() <= tolerance,
        "actual={actual}, expected={expected}, tolerance={tolerance}"
    );
}

fn close_vec3(actual: DVec3, expected: DVec3, tolerance: f64) {
    assert!(
        actual.distance(expected) <= tolerance,
        "actual={actual:?}, expected={expected:?}, tolerance={tolerance}"
    );
}

fn json_vec3(value: &Value) -> Result<DVec3, Box<dyn Error>> {
    Ok(DVec3::from_array(serde_json::from_value(value.clone())?))
}

fn compare_point(first: &DVec3, second: &DVec3) -> Ordering {
    first
        .x
        .total_cmp(&second.x)
        .then_with(|| first.y.total_cmp(&second.y))
        .then_with(|| first.z.total_cmp(&second.z))
}

fn compare_faces(first: &[DVec3], second: &[DVec3]) -> Ordering {
    for (first, second) in first.iter().zip(second) {
        let order = compare_point(first, second);
        if order != Ordering::Equal {
            return order;
        }
    }
    first.len().cmp(&second.len())
}

fn sort_points(points: &mut [DVec3]) {
    points.sort_by(compare_point);
}

fn sort_faces(faces: &mut [Vec<DVec3>]) {
    for face in faces.iter_mut() {
        sort_points(face);
    }
    faces.sort_by(|first, second| compare_faces(first, second));
}

fn json_points(value: &Value) -> Result<Vec<DVec3>, Box<dyn Error>> {
    let points: Vec<[f64; 3]> = serde_json::from_value(value.clone())?;
    let mut points: Vec<_> = points.into_iter().map(DVec3::from_array).collect();
    sort_points(&mut points);
    Ok(points)
}

fn json_faces(value: &Value) -> Result<Vec<Vec<DVec3>>, Box<dyn Error>> {
    let faces: Vec<Vec<[f64; 3]>> = serde_json::from_value(value.clone())?;
    let mut faces: Vec<Vec<DVec3>> = faces
        .into_iter()
        .map(|face| face.into_iter().map(DVec3::from_array).collect())
        .collect();
    sort_faces(&mut faces);
    Ok(faces)
}

fn mesh_faces_by_position(mesh: &Mesh) -> Result<Vec<Vec<DVec3>>, Box<dyn Error>> {
    let vertex_positions: HashMap<_, _> = mesh
        .vertices
        .iter()
        .map(|vertex| (vertex.id, vertex.co))
        .collect();
    let mut faces = Vec::with_capacity(mesh.faces.len());
    for face in &mesh.faces {
        let mut points = Vec::with_capacity(face.vertices.len());
        for id in &face.vertices {
            points.push(
                vertex_positions
                    .get(id)
                    .copied()
                    .ok_or("VDB mesh face references a missing vertex")?,
            );
        }
        faces.push(points);
    }
    sort_faces(&mut faces);
    Ok(faces)
}

fn assert_points_close(actual: &[DVec3], expected: &[DVec3], tolerance: f64) {
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected) {
        close_vec3(*actual, *expected, tolerance);
    }
}

fn assert_faces_close(actual: &[Vec<DVec3>], expected: &[Vec<DVec3>], tolerance: f64) {
    assert_eq!(actual.len(), expected.len());
    for (actual_face, expected_face) in actual.iter().zip(expected) {
        assert_points_close(actual_face, expected_face, tolerance);
    }
}

fn json_pair<T>(value: &Value) -> Result<(T, T), Box<dyn Error>>
where
    T: serde::de::DeserializeOwned,
{
    let pair: [T; 2] = serde_json::from_value(value.clone())?;
    let [minimum, maximum] = pair;
    Ok((minimum, maximum))
}

fn numeric(value: &Value) -> Result<f64, Box<dyn Error>> {
    if let Some(value) = value.as_f64() {
        return Ok(value);
    }
    if let Some(value) = value.as_bool() {
        return Ok(if value { 1.0 } else { 0.0 });
    }
    Err("OpenVDB oracle value is neither numeric nor boolean".into())
}

fn component(value: &Value, index: usize) -> Result<f64, Box<dyn Error>> {
    let value = value
        .as_array()
        .and_then(|values| values.get(index))
        .or_else(|| (index == 0 && !value.is_array()).then_some(value))
        .ok_or("OpenVDB oracle component is missing")?;
    numeric(value)
}

fn read_vdb(bytes: &[u8]) -> Result<VdbVolume, Box<dyn Error>> {
    Ok(VdbVolume::read(bytes)?)
}

fn file_volume(decoded: VdbVolume, grid_name: &str) -> VolumeData {
    VolumeData {
        source: VolumeSource::File(VolumeFileSource {
            format: "vdb".to_owned(),
            grid_names: vec![grid_name.to_owned()],
            ..VolumeFileSource::default()
        }),
        decoded_vdb: Some(Arc::new(decoded)),
        ..VolumeData::default()
    }
}

fn assert_volume_mesh_matches(
    volume: &VolumeData,
    grid_name: &str,
    oracle: &Value,
) -> Result<(), Box<dyn Error>> {
    let mesh = volume_to_mesh_named_adaptive(
        volume,
        oracle["threshold"]
            .as_f64()
            .ok_or("mesh threshold is missing")?,
        grid_name,
        oracle["adaptivity"]
            .as_f64()
            .ok_or("mesh adaptivity is missing")?,
    )?;
    let expected_vertices: usize = oracle["vertex_count"]
        .as_u64()
        .ok_or("mesh vertex count is missing")?
        .try_into()?;
    let expected_edges: usize = oracle["edge_count"]
        .as_u64()
        .ok_or("mesh edge count is missing")?
        .try_into()?;
    let expected_faces: usize = oracle["face_count"]
        .as_u64()
        .ok_or("mesh face count is missing")?
        .try_into()?;
    assert_eq!(mesh.vertices.len(), expected_vertices);
    assert_eq!(mesh.edges.len(), expected_edges);
    assert_eq!(mesh.faces.len(), expected_faces);
    let mut actual_vertices: Vec<_> = mesh.vertices.iter().map(|vertex| vertex.co).collect();
    sort_points(&mut actual_vertices);
    assert_points_close(&actual_vertices, &json_points(&oracle["vertices"])?, 1.0e-7);
    assert_faces_close(
        &mesh_faces_by_position(&mesh)?,
        &json_faces(&oracle["faces"])?,
        1.0e-7,
    );
    let bounds = mesh
        .bounds()
        .ok_or("VDB Volume to Mesh produced no vertices")?;
    let (expected_min, expected_max): (Value, Value) = json_pair(&oracle["bounds"])?;
    close_vec3(bounds.min, json_vec3(&expected_min)?, 1.0e-7);
    close_vec3(bounds.max, json_vec3(&expected_max)?, 1.0e-7);
    Ok(())
}

fn assert_world_bounds(
    grid: &potter_core::geom::vdb::VdbGrid,
    expected: &Value,
) -> Result<(), Box<dyn Error>> {
    let (expected_min, expected_max): (Value, Value) = json_pair(expected)?;
    let (actual_min, actual_max) = grid
        .active_bbox_world()
        .ok_or("VDB active world bounds are missing")?;
    close_vec3(actual_min, json_vec3(&expected_min)?, 1.0e-8);
    close_vec3(actual_max, json_vec3(&expected_max)?, 1.0e-8);
    Ok(())
}

#[test]
fn reads_blender_vdb_scalar_vector_and_integer_grid_types() -> Result<(), Box<dyn Error>> {
    let expected: Value = serde_json::from_str(EXPECTED)?;
    assert_eq!(expected["blender_version"], "5.2.2 LTS");
    assert_eq!(expected["openvdb_file_format_version"], 225);
    assert_eq!(
        expected["openvdb_library_version"],
        serde_json::json!([13, 0, 0])
    );
    assert_eq!(u32::from_le_bytes(VALUE_TYPES[8..12].try_into()?), 225);

    let volume = read_vdb(VALUE_TYPES)?;
    assert_eq!(volume.library_version, (13, 0));
    assert_eq!(
        volume.grids.len(),
        expected["value_types"]
            .as_array()
            .ok_or("typed grid oracle is missing")?
            .len()
    );

    let bpy_types = &expected["bpy_value_types"];
    let expected_names: Vec<_> = bpy_types["grid_names"]
        .as_array()
        .ok_or("Blender grid names are missing")?
        .iter()
        .map(|name| name.as_str().ok_or("Blender grid name is not a string"))
        .collect::<Result<_, _>>()?;
    let actual_names: Vec<_> = volume.grids.iter().map(|grid| grid.name.as_str()).collect();
    assert_eq!(actual_names, expected_names);

    for oracle in expected["value_types"]
        .as_array()
        .ok_or("typed grid oracle is missing")?
    {
        let name = oracle["name"].as_str().ok_or("grid name is missing")?;
        let grid = volume
            .grids
            .iter()
            .find(|grid| grid.name == name)
            .ok_or("typed grid was not imported")?;
        assert_eq!(grid.transform.map_type, "AffineMap");
        let expected_type = match name {
            "bool" => "bool",
            "int32" => "int32",
            "int64" => "int64",
            "vec3i" => "vec3i",
            "vec3d" | "half_vec3d" => "vec3d",
            "float" => "float",
            "double" | "half_double" => "double",
            "vec3s" => "vec3s",
            _ => return Err("unexpected typed grid name".into()),
        };
        let (bpy_type, channels) = match expected_type {
            "bool" => ("BOOLEAN", 1),
            "float" => ("FLOAT", 1),
            "double" => ("DOUBLE", 1),
            "int32" => ("INT", 1),
            "int64" => ("INT64", 1),
            "vec3s" => ("VECTOR_FLOAT", 3),
            "vec3d" => ("VECTOR_DOUBLE", 3),
            "vec3i" => ("VECTOR_INT", 3),
            _ => return Err("unexpected Blender VDB type".into()),
        };
        let bpy_grid = bpy_types["grids"]
            .as_array()
            .ok_or("Blender grid metadata is missing")?
            .iter()
            .find(|grid| grid["name"] == name)
            .ok_or("Blender did not expose the VDB grid")?;
        assert_eq!(bpy_grid["data_type"], bpy_type);
        assert_eq!(bpy_grid["channels"], channels);
        assert_eq!(grid.value_type.split('_').nth(1), Some(expected_type));
        let active_bbox: ([i32; 3], [i32; 3]) =
            serde_json::from_value(oracle["active_bbox"].clone())?;
        assert_eq!(grid.active_bbox, Some(active_bbox));
        assert_world_bounds(grid, &oracle["active_bbox_world"])?;

        for sample in oracle["samples"]
            .as_array()
            .ok_or("grid samples are missing")?
        {
            assert_eq!(sample["active"].as_bool(), Some(true));
            let index: [i32; 3] = serde_json::from_value(sample["index"].clone())?;
            let index_point = DVec3::from_array(index.map(f64::from));
            let world = json_vec3(&sample["world"])?;
            close_vec3(
                grid.transform.index_to_world.transform_point3(index_point),
                world,
                1.0e-9,
            );
            let value = &sample["value"];
            let component_count = value.as_array().map_or(1, Vec::len);
            for component_index in 0..component_count {
                let actual = grid
                    .sample_index_component(index, component_index)
                    .ok_or("VDB grid component is missing")?;
                assert_eq!(actual, component(value, component_index)?);
            }
            if !expected_type.starts_with("vec3") {
                let expected_value = numeric(value)?;
                let tolerance = 1.0e-8_f64.max(expected_value.abs() * 1.0e-12);
                close(grid.sample_world(world), expected_value, tolerance);
            }
        }

        let background_index: [i32; 3] =
            serde_json::from_value(oracle["background_sample"]["index"].clone())?;
        let background = &oracle["background_sample"]["value"];
        let component_count = background.as_array().map_or(1, Vec::len);
        for component_index in 0..component_count {
            let actual = grid
                .sample_index_component(background_index, component_index)
                .ok_or("VDB background component is missing")?;
            assert_eq!(actual, component(background, component_index)?);
        }
    }

    let mut future_version = VALUE_TYPES.to_vec();
    future_version[8..12].copy_from_slice(&226_u32.to_le_bytes());
    let Err(error) = VdbVolume::read(&future_version) else {
        return Err("future OpenVDB file versions must be rejected".into());
    };
    assert_eq!(error.code, ErrorCode::UnsupportedVersion);
    assert_eq!(
        error.details["feature_id"],
        "volume.openvdb_evaluation.file_version"
    );
    Ok(())
}

#[test]
fn samples_blender_nonlinear_frustum_map_and_bounds() -> Result<(), Box<dyn Error>> {
    let expected: Value = serde_json::from_str(EXPECTED)?;
    assert_eq!(u32::from_le_bytes(FRUSTUM[8..12].try_into()?), 225);
    let volume = read_vdb(FRUSTUM)?;
    assert_eq!(volume.grids.len(), 1);
    let grid = &volume.grids[0];
    let oracle = &expected["frustum"];
    assert_eq!(
        grid.name,
        oracle["name"].as_str().ok_or("frustum name is missing")?
    );
    assert_eq!(grid.transform.map_type, "NonlinearFrustumMap");
    let active_bbox: ([i32; 3], [i32; 3]) = serde_json::from_value(oracle["active_bbox"].clone())?;
    assert_eq!(grid.active_bbox, Some(active_bbox));
    assert_world_bounds(grid, &oracle["active_bbox_world"])?;

    for sample in oracle["samples"]
        .as_array()
        .ok_or("frustum samples are missing")?
    {
        let index: [i32; 3] = serde_json::from_value(sample["index"].clone())?;
        assert_eq!(
            grid.sample_index_component(index, 0),
            Some(numeric(&sample["value"])?)
        );
        let world = json_vec3(&sample["world"])?;
        close(grid.sample_world(world), numeric(&sample["value"])?, 1.0e-6);
    }
    let grid_sampler_case = oracle["samples"]
        .as_array()
        .and_then(|samples| samples.last())
        .ok_or("frustum sample oracle is missing")?;
    let world = json_vec3(&grid_sampler_case["world"])?;
    let grid_sampler_value = grid.sample_world(world);
    let affine_render_volume = file_volume(volume.clone(), "density");
    let rendered_linear_value = sample_density_checked_with(
        world,
        &affine_render_volume,
        potter_core::model::VolumeInterpolation::Linear,
    )?;
    assert!(
        (grid_sampler_value - rendered_linear_value).abs() > 1.0,
        "the full nonlinear GridSampler and renderer-affine frustum maps must be distinguished"
    );

    let bpy = &oracle["bpy_volume"];
    assert_eq!(bpy["grid_names"], serde_json::json!(["density"]));
    assert_eq!(bpy["grids"][0]["data_type"], "FLOAT");
    assert_eq!(bpy["grids"][0]["channels"], 1);
    let volume = file_volume(volume, "density");
    assert_volume_mesh_matches(&volume, "density", &bpy["volume_to_mesh"])?;
    Ok(())
}

#[test]
fn frustum_mesh_resampling_is_guarded_with_blender_oracle_cases() -> Result<(), Box<dyn Error>> {
    let expected: Value = serde_json::from_str(EXPECTED)?;
    let decoded = read_vdb(FRUSTUM)?;
    let volume = file_volume(decoded, "density");
    let cases = expected["frustum"]["bpy_resampled"]
        .as_array()
        .ok_or("frustum resampling oracle cases are missing")?;
    assert_eq!(cases.len(), 2);

    for case in cases {
        let mode = case["resolution_mode"]
            .as_str()
            .ok_or("frustum resolution mode is missing")?;
        assert!(
            case["vertex_count"]
                .as_u64()
                .is_some_and(|vertex_count| vertex_count > 0),
            "Blender's {mode} case must produce geometry"
        );
        let resolution = match mode {
            "VOXEL_SIZE" => VolumeToMeshResolution::VoxelSize(
                case["voxel_size"]
                    .as_f64()
                    .ok_or("frustum voxel size is missing")?,
            ),
            "VOXEL_AMOUNT" => VolumeToMeshResolution::VoxelAmount(
                case["voxel_amount"]
                    .as_u64()
                    .ok_or("frustum voxel amount is missing")?
                    .try_into()?,
            ),
            _ => return Err("unexpected frustum resolution mode".into()),
        };
        let Err(error) = volume_to_mesh_placed(
            &volume,
            0.5,
            "density",
            0.0,
            DMat4::IDENTITY,
            false,
            resolution,
        ) else {
            return Err(format!("frustum {mode} resampling should be refused").into());
        };
        assert_eq!(error.code, ErrorCode::UnsupportedFeature);
        assert_eq!(
            error.details["feature_id"],
            "volume.openvdb_evaluation.transform"
        );
        assert_eq!(error.details["resolution_mode"], mode);
    }
    Ok(())
}

#[test]
fn volume_to_mesh_consumes_blender_density_vdb() -> Result<(), Box<dyn Error>> {
    let expected: Value = serde_json::from_str(EXPECTED)?;
    assert_eq!(u32::from_le_bytes(DENSITY[8..12].try_into()?), 225);
    let decoded = read_vdb(DENSITY)?;
    assert_eq!(decoded.grids.len(), 1);
    let density_grid = decoded
        .density_grid()
        .ok_or("density VDB grid is missing")?;
    let density_expected = &expected["density"];
    let active_bbox: ([i32; 3], [i32; 3]) =
        serde_json::from_value(density_expected["active_bbox"].clone())?;
    assert_eq!(density_grid.active_bbox, Some(active_bbox));
    assert_world_bounds(density_grid, &density_expected["active_bbox_world"])?;

    let oracle = &expected["bpy_volume"];
    assert_eq!(oracle["grid_names"], serde_json::json!(["density"]));
    assert_eq!(oracle["grids"][0]["name"], "density");
    assert_eq!(oracle["grids"][0]["channels"], 1);
    assert_eq!(oracle["grids"][0]["data_type"], "FLOAT");
    let blender_matrix: [[f64; 4]; 4] =
        serde_json::from_value(oracle["grids"][0]["matrix_object"].clone())?;
    close_vec3(
        density_grid
            .transform
            .index_to_world
            .transform_point3(DVec3::ZERO),
        DVec3::new(
            blender_matrix[0][3],
            blender_matrix[1][3],
            blender_matrix[2][3],
        ),
        1.0e-9,
    );
    close_vec3(
        density_grid.voxel_size(),
        DVec3::new(
            blender_matrix[0][0],
            blender_matrix[1][1],
            blender_matrix[2][2],
        ),
        1.0e-9,
    );
    let volume = file_volume(decoded, "density");
    assert_volume_mesh_matches(&volume, "density", &oracle["volume_to_mesh"])?;
    Ok(())
}
