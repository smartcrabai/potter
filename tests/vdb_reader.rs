#![expect(clippy::unwrap_used, reason = "Blender integration fixture setup")]

use std::{
    env,
    error::Error,
    fs,
    io::BufReader,
    panic::AssertUnwindSafe,
    path::{Path, PathBuf},
    process::Command,
};

use glam::DVec3;
use potter::geom::vdb::VdbVolume;
use proptest::{prelude::*, test_runner::TestRunner};
use serde_json::{Value, json};
use tempfile::tempdir;

const FIXTURE_SCRIPT: &str = r#"
import bpy
import json
import math
import numpy as np
import openvdb
import os
import sys

root = os.path.realpath(sys.argv[sys.argv.index("--") + 1])

# Two sparse active values exercise negative indices and the standard Blosc writer.
density = openvdb.FloatGrid()
density.name = "density"
density_accessor = density.getAccessor()
density_accessor.setValueOn((-2, 0, 1), 1.25)
density_accessor.setValueOn((0, 0, 0), 0.5)
density_path = os.path.join(root, "density.vdb")
openvdb.write(density_path, grids=[density])

# The Python binding exposes saveFloatAsHalf on grids; values read back from the file
# are the reference after OpenVDB's half-float quantization.
half_grid = openvdb.FloatGrid()
half_grid.name = "half_density"
half_grid.saveFloatAsHalf = True
half_grid.getAccessor().setValueOn((2, 0, 0), 1.234567)
half_path = os.path.join(root, "half.vdb")
openvdb.write(half_path, grids=[half_grid])
half_expected = openvdb.read(half_path, "half_density").getAccessor().getValue((2, 0, 0))

vector_grid = openvdb.Vec3SGrid()
vector_grid.name = "velocity"
vector_grid.getAccessor().setValueOn((1, 2, 3), (3.0, 4.0, 12.0))
vector_path = os.path.join(root, "vector.vdb")
openvdb.write(vector_path, grids=[vector_grid])

# fill() creates a constant sparse tile instead of a dense leaf buffer.
tile_grid = openvdb.FloatGrid()
tile_grid.name = "tile_density"
tile_grid.fill((0, 0, 0), (7, 7, 7), 2.0, True)
tile_path = os.path.join(root, "tile.vdb")
openvdb.write(tile_path, grids=[tile_grid])

# Blender's Python transform API serializes a non-uniform scale-translate map.
scaled_grid = openvdb.FloatGrid()
scaled_grid.name = "scaled_density"
scaled_grid.getAccessor().setValueOn((1, 2, 3), 4.0)
scaled_grid.transform.postScale((2.0, 3.0, 4.0))
scaled_grid.transform.postTranslate((10.0, 20.0, 30.0))
scaled_path = os.path.join(root, "scaled.vdb")
openvdb.write(scaled_path, grids=[scaled_grid])
scaled_world = scaled_grid.transform.indexToWorld((1, 2, 3))

# A volume suitable for CLI inspect, beauty preview, and path rendering.
sphere = openvdb.FloatGrid()
sphere.name = "density"
dense = np.zeros((16, 16, 16), dtype=np.float32)
coordinates = np.indices(dense.shape) - 7.5
dense[(coordinates ** 2).sum(axis=0) < 36] = 1.0
sphere.copyFromArray(dense)
sphere_path = os.path.join(root, "sphere.vdb")
openvdb.write(sphere_path, grids=[sphere])
loaded_sphere = openvdb.read(sphere_path, "density")
minimum, maximum = loaded_sphere.evalActiveVoxelBoundingBox()

scene = bpy.context.scene
for obj in list(bpy.data.objects):
    bpy.data.objects.remove(obj, do_unlink=True)
volume_data = bpy.data.volumes.new("DensityVolume")
volume_data.filepath = "//sphere.vdb"
volume = bpy.data.objects.new("DensityVolume", volume_data)
scene.collection.objects.link(volume)
volume.location = (2.0, 0.0, 1.0)

camera_data = bpy.data.cameras.new("VolumeCamera")
camera_data.type = "ORTHO"
camera_data.ortho_scale = 24.0
camera = bpy.data.objects.new("VolumeCamera", camera_data)
scene.collection.objects.link(camera)
camera.location = (9.5, 7.5, 25.0)
scene.camera = camera
sun_data = bpy.data.lights.new("VolumeSun", "SUN")
sun_data.energy = 2.0
sun = bpy.data.objects.new("VolumeSun", sun_data)
scene.collection.objects.link(sun)
sun.rotation_mode = "QUATERNION"
sun.rotation_quaternion = (math.sqrt(0.5), 0.0, math.sqrt(0.5), 0.0)
scene.render.resolution_x = 64
scene.render.resolution_y = 64
scene.render.resolution_percentage = 100
scene.render.use_sequencer = False
scene.render.ffmpeg.audio_codec = "FLAC"
scene.render.image_settings.file_format = "PNG"
scene.render.engine = "CYCLES"
scene.cycles.device = "CPU"
scene.cycles.samples = 32
scene.cycles.seed = 17
scene.view_settings.view_transform = "Standard"
bpy.ops.wm.save_as_mainfile(
    filepath=os.path.join(root, "volume_no_material.blend"), relative_remap=True
)
scene.render.filepath = os.path.join(root, "cycles_no_material.png")
bpy.ops.render.render(write_still=True)
volume.hide_render = True
scene.render.filepath = os.path.join(root, "cycles_no_material_empty.png")
bpy.ops.render.render(write_still=True)
volume.hide_render = False

cloud_material = bpy.data.materials.new("CloudMaterial")
cloud_material.use_nodes = True
cloud_material.node_tree.nodes.clear()
volume_shader = cloud_material.node_tree.nodes.new("ShaderNodeVolumePrincipled")
volume_shader.inputs["Density"].default_value = 1.0
volume_shader.inputs["Color"].default_value = (0.5, 0.5, 0.5, 1.0)
volume_shader.inputs["Anisotropy"].default_value = 0.0
material_output = cloud_material.node_tree.nodes.new("ShaderNodeOutputMaterial")
cloud_material.node_tree.links.new(
    volume_shader.outputs["Volume"], material_output.inputs["Volume"]
)
volume.data.materials.append(cloud_material)

blend_path = os.path.join(root, "volume.blend")
bpy.ops.wm.save_as_mainfile(filepath=blend_path, relative_remap=True)
scene.render.filepath = os.path.join(root, "cycles_volume.png")
bpy.ops.render.render(write_still=True)
volume.hide_render = True
scene.render.filepath = os.path.join(root, "cycles_empty.png")
bpy.ops.render.render(write_still=True)

expected = {
    "half": float(half_expected),
    "vector": [3.0, 4.0, 12.0],
    "scaled_world": list(scaled_world),
    "sphere_min": [float(value) for value in minimum],
    "sphere_max": [float(value) for value in maximum],
}
with open(os.path.join(root, "expected.json"), "w", encoding="utf-8") as output:
    json.dump(expected, output)
"#;

const MANTA_BAKE_SCRIPT: &str = r#"
import bpy
import glob
import json
import openvdb
import os
import sys

root = os.path.realpath(sys.argv[sys.argv.index("--") + 1])
rows = []
for compression in ("NONE", "ZIP", "BLOSC"):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    cache_directory = os.path.join(root, "cache", compression.lower())
    os.makedirs(cache_directory, exist_ok=True)

    bpy.ops.mesh.primitive_cube_add(size=4.0)
    domain = bpy.context.object
    domain.name = "SmokeDomain"
    domain_modifier = domain.modifiers.new("FluidDomain", "FLUID")
    domain_modifier.fluid_type = "DOMAIN"
    settings = domain_modifier.domain_settings
    settings.domain_type = "GAS"
    settings.resolution_max = 8
    settings.cache_type = "MODULAR"
    settings.cache_directory = cache_directory
    settings.cache_frame_start = 1
    settings.cache_frame_end = 2
    settings.cache_data_format = "OPENVDB"
    settings.openvdb_cache_compress_type = compression
    depth_property = settings.bl_rna.properties.get("openvdb_data_depth")
    if depth_property is not None:
        depth_options = {item.identifier for item in depth_property.enum_items}
        if "32" in depth_options:
            settings.openvdb_data_depth = "32"
        elif "16" in depth_options:
            settings.openvdb_data_depth = "16"

    bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=1, radius=0.5)
    flow = bpy.context.object
    flow.name = "SmokeFlow"
    flow_modifier = flow.modifiers.new("FluidFlow", "FLUID")
    flow_modifier.fluid_type = "FLOW"
    flow_modifier.flow_settings.flow_type = "SMOKE"
    flow_modifier.flow_settings.flow_behavior = "INFLOW"
    flow_modifier.flow_settings.density = 1.0

    for obj in bpy.context.selected_objects:
        obj.select_set(False)
    domain.select_set(True)
    bpy.context.view_layer.objects.active = domain
    with bpy.context.temp_override(
        object=domain,
        active_object=domain,
        selected_objects=[domain],
        selected_editable_objects=[domain],
    ):
        assert bpy.ops.fluid.bake_data() == {"FINISHED"}

    for frame in (1, 2):
        pattern = os.path.join(cache_directory, "**", f"fluid_data_{frame:04d}.vdb")
        matches = glob.glob(pattern, recursive=True)
        if len(matches) != 1:
            raise RuntimeError(f"expected one frame {frame} VDB for {compression}, got {matches}")
        path = matches[0]
        grid = openvdb.read(path, "density")
        accessor = grid.getAccessor()
        minimum, maximum = grid.evalActiveVoxelBoundingBox()
        center = tuple((minimum[axis] + maximum[axis]) // 2 for axis in range(3))
        indices = [minimum, center, maximum, tuple(value - 1 for value in minimum)]
        rows.append({
            "compression": compression,
            "frame": frame,
            "path": os.path.relpath(path, root),
            "bbox_min": [int(value) for value in minimum],
            "bbox_max": [int(value) for value in maximum],
            "samples": [
                {"index": [int(value) for value in index], "value": float(accessor.getValue(index))}
                for index in indices
            ],
        })

with open(os.path.join(root, "mantaflow_expected.json"), "w", encoding="utf-8") as output:
    json.dump(rows, output)
"#;

fn blender_executable() -> Option<PathBuf> {
    fn usable(path: PathBuf) -> Option<PathBuf> {
        path.is_file().then_some(path)
    }
    if let Some(path) = env::var_os("POTTER_BLENDER") {
        return usable(PathBuf::from(path));
    }
    if let Some(path) = env::split_paths(&env::var_os("PATH")?)
        .map(|directory| directory.join("blender"))
        .find_map(usable)
    {
        return Some(path);
    }
    usable(PathBuf::from(
        "/Applications/Blender.app/Contents/MacOS/Blender",
    ))
}

fn run_ok(command: &mut Command, label: &str) -> Result<std::process::Output, Box<dyn Error>> {
    let output = command.output()?;
    if !output.status.success() {
        return Err(std::io::Error::other(format!(
            "{label} failed: stdout={} stderr={}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        ))
        .into());
    }
    Ok(output)
}

fn make_fixture(blender: &Path, root: &Path) -> Result<PathBuf, Box<dyn Error>> {
    let script_path = root.join("make_vdb.py");
    fs::write(&script_path, FIXTURE_SCRIPT)?;
    let canonical_root = fs::canonicalize(root)?;
    let mut command = Command::new(blender);
    command
        .args(["--background", "--factory-startup", "--python"])
        .arg(script_path)
        .args(["--", canonical_root.to_str().unwrap()]);
    run_ok(&mut command, "Blender VDB fixture generation")?;
    Ok(root.join("density.vdb"))
}

fn make_mantaflow_fixture(blender: &Path, root: &Path) -> Result<PathBuf, Box<dyn Error>> {
    let script_path = root.join("make_mantaflow_vdb.py");
    fs::write(&script_path, MANTA_BAKE_SCRIPT)?;
    let canonical_root = fs::canonicalize(root)?;
    let mut command = Command::new(blender);
    command
        .args(["--background", "--factory-startup", "--python"])
        .arg(script_path)
        .args(["--", canonical_root.to_str().unwrap()]);
    run_ok(&mut command, "Blender Mantaflow VDB bake")?;
    Ok(root.join("mantaflow_expected.json"))
}

fn pot_json(args: &[&str], label: &str) -> Result<Value, Box<dyn Error>> {
    let mut command = Command::new(env!("CARGO_BIN_EXE_pot"));
    command.args(args).arg("--json");
    let output = run_ok(&mut command, label)?;
    Ok(serde_json::from_slice(&output.stdout)?)
}

fn assert_close(actual: f64, expected: f64, tolerance: f64) {
    assert!(
        (actual - expected).abs() <= tolerance,
        "actual={actual}, expected={expected}, tolerance={tolerance}"
    );
}

fn read_png(path: &Path) -> Result<(u32, u32, Vec<u8>), Box<dyn Error>> {
    let decoder = png::Decoder::new(BufReader::new(fs::File::open(path)?));
    let mut reader = decoder.read_info()?;
    let mut bytes = vec![
        0;
        reader
            .output_buffer_size()
            .ok_or("PNG output buffer size is unknown")?
    ];
    let info = reader.next_frame(&mut bytes)?;
    bytes.truncate(info.buffer_size());
    Ok((info.width, info.height, bytes))
}
fn assert_cycles_default_volume(root: &Path) -> Result<(), Box<dyn Error>> {
    let (width, height, default_pixels) = read_png(&root.join("cycles_no_material.png"))?;
    let (principled_width, principled_height, principled_pixels) =
        read_png(&root.join("cycles_volume.png"))?;
    assert_eq!((width, height), (principled_width, principled_height));
    let channels = default_pixels.len() / (usize::try_from(width)? * usize::try_from(height)?);
    let center = ((usize::try_from(height)? / 2) * usize::try_from(width)?
        + usize::try_from(width)? / 2)
        * channels;
    let default_center = &default_pixels[center..center + 3];
    let principled_center = &principled_pixels[center..center + 3];
    assert!(
        (0..3).all(|channel| { default_center[channel].abs_diff(principled_center[channel]) <= 4 }),
        "Cycles no-material center={default_center:?}, Principled center={principled_center:?}"
    );
    assert!(
        default_pixels[center..center + 3]
            .iter()
            .any(|channel| *channel > 0),
        "Cycles' no-material Volume default must not render as an opaque black object"
    );
    Ok(())
}

fn assert_volume_pixels(path: &Path) -> Result<(), Box<dyn Error>> {
    let (width, height, pixels) = read_png(path)?;
    assert!(width > 2 && height > 2);
    let pixel_count = usize::try_from(width)?
        .checked_mul(usize::try_from(height)?)
        .ok_or("PNG pixel count overflows usize")?;
    let channel_count = pixels.len() / pixel_count;
    assert!(channel_count >= 3 && pixels.len() % pixel_count == 0);
    let background = &pixels[..3];
    let width = usize::try_from(width)?;
    let height = usize::try_from(height)?;
    let mut projected_changes = 0_usize;
    for y in height / 4..height * 3 / 4 {
        for x in width / 4..width * 3 / 4 {
            let offset = (y * width + x) * channel_count;
            if &pixels[offset..offset + 3] != background {
                projected_changes += 1;
            }
        }
    }
    assert!(
        projected_changes > 0,
        "{} has no non-background pixels in the projected volume region",
        path.display()
    );
    Ok(())
}

fn assert_volume_disc(
    volume_path: &Path,
    empty_path: &Path,
    center_x: f64,
    center_y: f64,
    radius: f64,
) -> Result<(), Box<dyn Error>> {
    let (width, height, volume_pixels) = read_png(volume_path)?;
    let (empty_width, empty_height, empty_pixels) = read_png(empty_path)?;
    assert_eq!((width, height), (empty_width, empty_height));
    let pixel_count = usize::try_from(width)?
        .checked_mul(usize::try_from(height)?)
        .ok_or("PNG pixel count overflows usize")?;
    let channels = volume_pixels.len() / pixel_count;
    assert!(channels >= 3 && volume_pixels.len() % pixel_count == 0);
    assert_eq!(empty_pixels.len(), volume_pixels.len());
    let mut covered = 0_usize;
    let mut centroid_x = 0.0;
    let mut centroid_y = 0.0;
    let width = usize::try_from(width)?;
    let height = usize::try_from(height)?;
    for y in 0..height {
        for x in 0..width {
            let offset = (y * width + x) * channels;
            let changed = (0..3).any(|channel| {
                volume_pixels[offset + channel].abs_diff(empty_pixels[offset + channel]) > 2
            });
            if changed {
                covered += 1;
                centroid_x += f64::from(u32::try_from(x)?);
                centroid_y += f64::from(u32::try_from(y)?);
            }
        }
    }
    for (x, y) in [
        (0, 0),
        (width - 1, 0),
        (0, height - 1),
        (width - 1, height - 1),
    ] {
        let offset = (y * width + x) * channels;
        assert_eq!(
            &volume_pixels[offset..offset + 3],
            &empty_pixels[offset..offset + 3],
            "volume corner pixel ({x}, {y}) differs from the empty-scene background"
        );
    }
    let center_offset = ((height / 2) * width + width / 2) * channels;
    assert!(
        (0..3).any(|channel| {
            volume_pixels[center_offset + channel].abs_diff(empty_pixels[center_offset + channel])
                > 2
        }),
        "volume center pixel matches the empty-scene background"
    );
    let expected_area = std::f64::consts::PI * radius * radius;
    let covered_f64 = f64::from(u32::try_from(covered)?);
    let area_error = (covered_f64 - expected_area).abs() / expected_area;
    assert!(
        area_error <= 0.2,
        "{} covered pixels do not approximate a disc of radius {radius:.2}: \
         area={covered}, expected={expected_area:.1}, relative_error={area_error:.3}",
        volume_path.display()
    );
    let centroid_x = centroid_x / covered_f64;
    let centroid_y = centroid_y / covered_f64;
    assert!(
        (centroid_x - center_x).hypot(centroid_y - center_y) <= 2.0,
        "{} projected centroid ({centroid_x:.2}, {centroid_y:.2}) is more than 2 pixels \
         from sphere center ({center_x:.2}, {center_y:.2})",
        volume_path.display()
    );
    let center_rgb = &volume_pixels[center_offset..center_offset + 3];
    let background_rgb = &empty_pixels[center_offset..center_offset + 3];
    let center_luminance = 0.2126 * f64::from(center_rgb[0])
        + 0.7152 * f64::from(center_rgb[1])
        + 0.0722 * f64::from(center_rgb[2]);
    let background_luminance = 0.2126 * f64::from(background_rgb[0])
        + 0.7152 * f64::from(background_rgb[1])
        + 0.0722 * f64::from(background_rgb[2]);
    assert!(
        center_luminance > 2.0,
        "{} volume center must remain visibly lit rather than black: \
         luminance={center_luminance:.1}, background={background_luminance:.1}",
        volume_path.display()
    );
    eprintln!(
        "{}: covered={covered}, expected_area={expected_area:.1}, \
         centroid=({centroid_x:.2},{centroid_y:.2}), center=({center_x:.2},{center_y:.2}), \
         center_rgb={center_rgb:?}, background_rgb={background_rgb:?}",
        volume_path.display()
    );
    Ok(())
}
fn linear_luminance(pixels: &[u8], offset: usize) -> f64 {
    let channel = |index: usize| {
        let encoded = f64::from(pixels[offset + index]) / 255.0;
        if encoded <= 0.04045 {
            encoded / 12.92
        } else {
            ((encoded + 0.055) / 1.055).powf(2.4)
        }
    };
    0.2126 * channel(0) + 0.7152 * channel(1) + 0.0722 * channel(2)
}

fn compare_cycles_and_path_volume(
    path_volume: &Path,
    path_empty: &Path,
    cycles_volume: &Path,
    cycles_empty: &Path,
) -> Result<(), Box<dyn Error>> {
    let (width, height, path_pixels) = read_png(path_volume)?;
    let (empty_width, empty_height, path_background) = read_png(path_empty)?;
    let (cycles_width, cycles_height, cycles_pixels) = read_png(cycles_volume)?;
    let (cycles_empty_width, cycles_empty_height, cycles_background) = read_png(cycles_empty)?;
    assert_eq!((width, height), (empty_width, empty_height));
    assert_eq!((width, height), (cycles_width, cycles_height));
    assert_eq!((width, height), (cycles_empty_width, cycles_empty_height));
    let channels = path_pixels.len() / (usize::try_from(width)? * usize::try_from(height)?);
    let mut count = 0_usize;
    let mut path_contribution = 0.0;
    let mut cycles_contribution = 0.0;
    let mut path_left = 0.0;
    let mut path_right = 0.0;
    let mut cycles_left = 0.0;
    let mut cycles_right = 0.0;
    let mut left_count = 0_usize;
    let mut right_count = 0_usize;
    for y in 0..usize::try_from(height)? {
        for x in 0..usize::try_from(width)? {
            let offset = (y * usize::try_from(width)?) * channels + x * channels;
            let changed = (0..3).any(|channel| {
                path_pixels[offset + channel].abs_diff(path_background[offset + channel]) > 2
            });
            if !changed {
                continue;
            }
            let path_delta =
                linear_luminance(&path_pixels, offset) - linear_luminance(&path_background, offset);
            let cycles_delta = linear_luminance(&cycles_pixels, offset)
                - linear_luminance(&cycles_background, offset);
            count += 1;
            path_contribution += path_delta;
            cycles_contribution += cycles_delta;
            if (x as f64) < f64::from(width) * 0.5 {
                path_left += path_delta;
                cycles_left += cycles_delta;
                left_count += 1;
            } else {
                path_right += path_delta;
                cycles_right += cycles_delta;
                right_count += 1;
            }
        }
    }
    assert!(count > 0, "the path-rendered VDB must cover pixels");
    let mean_path = path_contribution / count as f64;
    let mean_cycles = cycles_contribution / count as f64;
    let relative_error = (mean_path - mean_cycles).abs() / mean_cycles.abs().max(1.0e-6);
    let path_order = if left_count == 0 || right_count == 0 {
        0.0
    } else {
        path_right / right_count as f64 - path_left / left_count as f64
    };
    let cycles_order = if left_count == 0 || right_count == 0 {
        0.0
    } else {
        cycles_right / right_count as f64 - cycles_left / left_count as f64
    };
    eprintln!(
        "Cycles/path VDB sphere: mean contribution cycles={mean_cycles:.5}, path={mean_path:.5}, \
         relative error={relative_error:.1}%, screen-side deltas cycles={cycles_order:.5}, path={path_order:.5}",
        relative_error = relative_error * 100.0
    );
    assert!(
        relative_error <= 0.25,
        "Cycles/path mean VDB luminance differs by {percent:.1}% (>25%)",
        percent = relative_error * 100.0
    );
    assert!(
        path_order * cycles_order > 0.0,
        "Cycles and path renders must have the same lit/shadow side ordering: \
         cycles={cycles_order}, path={path_order}"
    );
    Ok(())
}

fn copy_project(source: &Path, destination: &Path) -> Result<(), Box<dyn Error>> {
    fs::create_dir_all(destination)?;
    for entry in fs::read_dir(source)? {
        let entry = entry?;
        let target = destination.join(entry.file_name());
        if entry.file_type()?.is_dir() {
            copy_project(&entry.path(), &target)?;
        } else if entry.file_type()?.is_file() {
            fs::copy(entry.path(), target)?;
        }
    }
    Ok(())
}

fn read_vdb(path: &Path, label: &str) -> Result<VdbVolume, Box<dyn Error>> {
    let bytes = fs::read(path)?;
    VdbVolume::read(&bytes)
        .map_err(|error| std::io::Error::other(format!("could not read {label}: {error}")).into())
}
fn read_index(value: &Value) -> Result<[i32; 3], Box<dyn Error>> {
    Ok(serde_json::from_value(value.clone())?)
}

#[test]
fn reads_blender_float_half_vector_tile_and_nonuniform_transform_grids()
-> Result<(), Box<dyn Error>> {
    let Some(blender) = blender_executable() else {
        eprintln!("skipping VDB reader integration; no Blender executable was found");
        return Ok(());
    };
    let directory = tempdir()?;
    let density_path = make_fixture(&blender, directory.path())?;
    let expected: Value =
        serde_json::from_slice(&fs::read(directory.path().join("expected.json"))?)?;

    let density = read_vdb(&density_path, "density grid")?;
    let grid = density.density_grid().ok_or("density grid was not read")?;
    assert_close(f64::from(grid.sample_index([0, 0, 0])), 0.5, 1.0e-6);
    assert_close(f64::from(grid.sample_index([-2, 0, 1])), 1.25, 1.0e-6);
    assert_close(
        f64::from(grid.sample_index([10, 10, 10])),
        f64::from(grid.tree.background),
        1.0e-6,
    );
    assert_eq!(grid.active_bbox, Some(([-2, 0, 0], [0, 0, 1])));

    let half = read_vdb(&directory.path().join("half.vdb"), "half-float grid")?;
    let half_grid = half.density_grid().ok_or("half-float grid was not read")?;
    assert_close(
        f64::from(half_grid.sample_index([2, 0, 0])),
        expected["half"]
            .as_f64()
            .ok_or("half expected value is missing")?,
        1.0e-6,
    );

    let vector = read_vdb(&directory.path().join("vector.vdb"), "vector grid")?;
    let vector_grid = vector.density_grid().ok_or("vector grid was not read")?;
    assert_eq!(vector_grid.sample_index_component([1, 2, 3], 0), Some(3.0));
    assert_eq!(vector_grid.sample_index_component([1, 2, 3], 1), Some(4.0));
    assert_eq!(vector_grid.sample_index_component([1, 2, 3], 2), Some(12.0));
    assert_eq!(vector_grid.sample_index_component([1, 2, 3], 3), None);
    assert_close(f64::from(vector_grid.sample_index([1, 2, 3])), 13.0, 1.0e-6);
    let sphere = read_vdb(&directory.path().join("sphere.vdb"), "dense sphere grid")?;
    let sphere_grid = sphere
        .density_grid()
        .ok_or("sphere density grid was not read")?;
    assert_close(f64::from(sphere_grid.sample_index([7, 7, 7])), 1.0, 1.0e-6);
    assert_close(f64::from(sphere_grid.sample_index([0, 0, 0])), 0.0, 1.0e-6);
    assert_close(
        sphere_grid.sample_world(DVec3::new(2.0, 2.0, 7.5)),
        0.0,
        1.0e-6,
    );
    assert_close(sphere_grid.sample_world(DVec3::splat(7.5)), 1.0, 1.0e-6);

    let tile = read_vdb(&directory.path().join("tile.vdb"), "tile grid")?;
    let tile_grid = tile.density_grid().ok_or("tile grid was not read")?;
    assert_close(f64::from(tile_grid.sample_index([7, 7, 7])), 2.0, 1.0e-6);
    assert_close(
        f64::from(tile_grid.sample_index([8, 8, 8])),
        f64::from(tile_grid.tree.background),
        1.0e-6,
    );
    assert_eq!(tile_grid.active_bbox, Some(([0, 0, 0], [7, 7, 7])));

    let scaled = read_vdb(&directory.path().join("scaled.vdb"), "scaled grid")?;
    let scaled_grid = scaled.density_grid().ok_or("scaled grid was not read")?;
    let world = DVec3::from_array(
        expected["scaled_world"]
            .as_array()
            .ok_or("scaled world position is missing")?
            .iter()
            .map(|value| value.as_f64().unwrap())
            .collect::<Vec<_>>()
            .try_into()
            .map_err(|_| "scaled world position must have three components")?,
    );
    assert_close(scaled_grid.sample_world(world), 4.0, 1.0e-6);
    let (world_min, world_max) = scaled_grid
        .active_bbox_world()
        .ok_or("scaled grid active bounds are missing")?;
    assert_eq!(world_min, world);
    assert_eq!(world_max, world);

    // Blenders' Python API exposes only default-compression openvdb.write overloads.
    let bytes = fs::read(directory.path().join("density.vdb"))?;
    let mut runner = TestRunner::default();
    let strategy = (0_usize..=bytes.len(), 0_usize..bytes.len(), any::<u8>());
    runner.run(&strategy, |(end, index, flipped)| {
        let truncated =
            std::panic::catch_unwind(AssertUnwindSafe(|| VdbVolume::read(&bytes[..end])));
        prop_assert!(
            truncated.is_ok(),
            "reader panicked on prefix of length {end}"
        );
        let mut changed = bytes.clone();
        changed[index] ^= flipped;
        let corrupted = std::panic::catch_unwind(AssertUnwindSafe(|| VdbVolume::read(&changed)));
        prop_assert!(
            corrupted.is_ok(),
            "reader panicked with mutation at {index}"
        );
        Ok(())
    })?;
    Ok(())
}

#[test]
fn mantaflow_bakes_raw_zip_and_blosc_vdb_for_two_frames() -> Result<(), Box<dyn Error>> {
    let Some(blender) = blender_executable() else {
        eprintln!("skipping Mantaflow VDB integration; no Blender executable was found");
        return Ok(());
    };
    let directory = tempdir()?;
    let expected_path = make_mantaflow_fixture(&blender, directory.path())?;
    let expected: Value = serde_json::from_slice(&fs::read(expected_path)?)?;
    let rows = expected
        .as_array()
        .ok_or("Mantaflow expected samples are missing")?;
    assert_eq!(
        rows.len(),
        6,
        "expected three compression modes across two frames"
    );
    for compression in ["NONE", "ZIP", "BLOSC"] {
        assert!(
            rows.iter()
                .any(|row| row["compression"].as_str() == Some(compression)),
            "Mantaflow fixture omitted {compression}"
        );
    }

    for row in rows {
        let relative_path = row["path"]
            .as_str()
            .ok_or("Mantaflow output path is missing")?;
        let path = directory.path().join(relative_path);
        let label = format!(
            "{} frame {}",
            row["compression"].as_str().unwrap_or("unknown"),
            row["frame"].as_u64().unwrap_or_default()
        );
        let volume = read_vdb(&path, &label)?;
        let grid = volume
            .density_grid()
            .ok_or("baked VDB density grid was not read")?;
        let active_min: [i32; 3] = serde_json::from_value(row["bbox_min"].clone())?;
        let active_max: [i32; 3] = serde_json::from_value(row["bbox_max"].clone())?;
        assert_eq!(grid.active_bbox, Some((active_min, active_max)), "{label}");
        let samples = row["samples"]
            .as_array()
            .ok_or("Mantaflow sample list is missing")?;
        assert!(
            samples
                .iter()
                .any(|sample| { sample["value"].as_f64().is_some_and(|value| value > 0.0) }),
            "{label} contains no active density samples"
        );
        for sample in samples {
            let index = read_index(&sample["index"])?;
            let expected_value = sample["value"]
                .as_f64()
                .ok_or("Mantaflow expected value is missing")?;
            assert_close(f64::from(grid.sample_index(index)), expected_value, 1.0e-6);
        }
    }
    Ok(())
}

#[test]
fn imports_vdb_resources_and_renders_real_volume_data() -> Result<(), Box<dyn Error>> {
    let Some(blender) = blender_executable() else {
        eprintln!("skipping VDB CLI integration; no Blender executable was found");
        return Ok(());
    };
    let directory = tempdir()?;
    let _density_path = make_fixture(&blender, directory.path())?;
    assert_cycles_default_volume(directory.path())?;
    let project = directory.path().join("project");
    assert_eq!(
        pot_json(
            &["init", project.to_str().unwrap()],
            "project initialization"
        )?["ok"],
        true
    );
    let imported = pot_json(
        &[
            "import",
            project.to_str().unwrap(),
            "--file",
            directory.path().join("volume.blend").to_str().unwrap(),
            "--format",
            "blend",
            "--mode",
            "replace",
            "--base-revision",
            "0",
            "--blender",
            blender.to_str().unwrap(),
        ],
        "VDB Blender import",
    )?;
    assert_eq!(imported["ok"], true);

    let document: Value = serde_json::from_slice(&fs::read(project.join("scene.json"))?)?;
    let (volume_id, _) = document["nodes"]
        .as_object()
        .and_then(|nodes| nodes.iter().find(|(_, node)| node["kind"] == "volume"))
        .ok_or("imported volume node is missing")?;
    let inspect = pot_json(
        &["inspect", project.to_str().unwrap(), "--id", volume_id],
        "VDB bounds inspection",
    )?;
    let item = &inspect["result"]["items"][0];
    let volume_offset = DVec3::new(2.0, 0.0, 1.0);
    let expected_min = DVec3::from_array(serde_json::from_value(expected_bounds(
        directory.path(),
        "sphere_min",
    )?)?);
    let expected_max = DVec3::from_array(serde_json::from_value(expected_bounds(
        directory.path(),
        "sphere_max",
    )?)?);
    let inspected_min = DVec3::from_array(serde_json::from_value(item["bounds"]["min"].clone())?);
    let inspected_max = DVec3::from_array(serde_json::from_value(item["bounds"]["max"].clone())?);
    for axis in 0..3 {
        assert_close(
            inspected_min[axis],
            expected_min[axis] + volume_offset[axis],
            1.0e-6,
        );
        assert_close(
            inspected_max[axis],
            expected_max[axis] + volume_offset[axis],
            1.0e-6,
        );
    }

    let revision = document["revision"]
        .as_u64()
        .ok_or("imported scene revision is missing")?;
    let empty_project = directory.path().join("project_without_volume");
    copy_project(&project, &empty_project)?;
    let delete_volume_path = directory.path().join("delete_volume.json");
    fs::write(
        &delete_volume_path,
        serde_json::to_vec(&json!({
            "schema_version":1,
            "base_revision":revision,
            "operations":[{"op":"node.delete","target":{"id":volume_id}}]
        }))?,
    )?;
    let deleted_volume = pot_json(
        &[
            "apply",
            empty_project.to_str().unwrap(),
            "--file",
            delete_volume_path.to_str().unwrap(),
        ],
        "empty-scene volume removal",
    )?;
    assert_eq!(deleted_volume["ok"], true);

    let beauty_dir = directory.path().join("beauty");
    let beauty = pot_json(
        &[
            "preview",
            project.to_str().unwrap(),
            "--views",
            "front",
            "--size",
            "64",
            "--mode",
            "beauty",
            "--out",
            beauty_dir.to_str().unwrap(),
        ],
        "VDB beauty preview",
    )?;
    assert_eq!(beauty["ok"], true);
    let empty_beauty_dir = directory.path().join("empty_beauty");
    let empty_beauty = pot_json(
        &[
            "preview",
            empty_project.to_str().unwrap(),
            "--views",
            "front",
            "--size",
            "64",
            "--mode",
            "beauty",
            "--out",
            empty_beauty_dir.to_str().unwrap(),
        ],
        "empty-scene beauty preview",
    )?;
    assert_eq!(empty_beauty["ok"], true);
    assert_volume_disc(
        &beauty_dir.join("front.png"),
        &empty_beauty_dir.join("front.png"),
        31.5,
        31.5,
        6.0 * 64.0 / 12.1,
    )?;

    for mode in ["solid", "wire"] {
        let preview_dir = directory.path().join(format!("{mode}_preview"));
        let preview = pot_json(
            &[
                "preview",
                project.to_str().unwrap(),
                "--views",
                "iso",
                "--size",
                "64",
                "--mode",
                mode,
                "--out",
                preview_dir.to_str().unwrap(),
            ],
            &format!("VDB {mode} preview"),
        )?;
        assert_eq!(preview["ok"], true);
        assert_volume_pixels(&preview_dir.join("iso.png"))?;
    }

    for engine in ["path", "realtime"] {
        let render_dir = directory.path().join(format!("{engine}_render"));
        let rendered = pot_json(
            &[
                "render",
                project.to_str().unwrap(),
                "--engine",
                engine,
                "--format",
                "png",
                "--out",
                render_dir.to_str().unwrap(),
            ],
            &format!("VDB {engine} render"),
        )?;
        assert_eq!(rendered["ok"], true);
        let image = fs::read_dir(&render_dir)?
            .filter_map(Result::ok)
            .map(|entry| entry.path())
            .find(|path| path.extension().is_some_and(|extension| extension == "png"))
            .ok_or("renderer did not create a PNG")?;

        let empty_render_dir = directory.path().join(format!("empty_{engine}_render"));
        let empty_rendered = pot_json(
            &[
                "render",
                empty_project.to_str().unwrap(),
                "--engine",
                engine,
                "--format",
                "png",
                "--out",
                empty_render_dir.to_str().unwrap(),
            ],
            &format!("empty-scene {engine} render"),
        )?;
        assert_eq!(empty_rendered["ok"], true);
        let empty_image = fs::read_dir(&empty_render_dir)?
            .filter_map(Result::ok)
            .map(|entry| entry.path())
            .find(|path| path.extension().is_some_and(|extension| extension == "png"))
            .ok_or("renderer did not create the empty-scene PNG")?;
        assert_volume_disc(&image, &empty_image, 31.5, 31.5, 16.0)?;
        if engine == "path" {
            compare_cycles_and_path_volume(
                &image,
                &empty_image,
                &directory.path().join("cycles_volume.png"),
                &directory.path().join("cycles_empty.png"),
            )?;
        }
    }

    let operations_path = directory.path().join("vdb_volume_to_mesh.json");
    fs::write(
        &operations_path,
        serde_json::to_vec(&json!({
            "schema_version":1,
            "base_revision":revision,
            "operations":[
                {"op":"node.create","id":"vdb_surface","kind":"box","params":{}},
                {"op":"modifier.create","target":{"id":"vdb_surface"},"id":"extract","type":"volume_to_mesh","params":{"object":volume_id,"threshold":0.5}}
            ]
        }))?,
    )?;
    let applied = pot_json(
        &[
            "apply",
            project.to_str().unwrap(),
            "--file",
            operations_path.to_str().unwrap(),
        ],
        "VDB volume-to-mesh modifier",
    )?;
    assert_eq!(applied["ok"], true);
    let surface = pot_json(
        &["inspect", project.to_str().unwrap(), "--id", "vdb_surface"],
        "VDB volume-to-mesh bounds",
    )?;
    let surface_bounds = &surface["result"]["items"][0]["bounds"];
    let mesh_min = DVec3::from_array(serde_json::from_value(surface_bounds["min"].clone())?);
    let mesh_max = DVec3::from_array(serde_json::from_value(surface_bounds["max"].clone())?);
    let active_min = DVec3::from_array(serde_json::from_value(expected_bounds(
        directory.path(),
        "sphere_min",
    )?)?);
    let active_max = DVec3::from_array(serde_json::from_value(expected_bounds(
        directory.path(),
        "sphere_max",
    )?)?);
    for axis in 0..3 {
        assert_close(mesh_min[axis], active_min[axis] + volume_offset[axis], 1.0);
        assert_close(mesh_max[axis], active_max[axis] + volume_offset[axis], 1.0);
    }
    Ok(())
}

fn expected_bounds(root: &Path, key: &str) -> Result<Value, Box<dyn Error>> {
    let expected: Value = serde_json::from_slice(&fs::read(root.join("expected.json"))?)?;
    Ok(Value::Array(
        expected[key]
            .as_array()
            .ok_or("expected bounds are missing")?
            .clone(),
    ))
}
