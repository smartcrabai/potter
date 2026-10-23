//! Sun, point, and spot emitters, and an area light through absorbing voxel and mesh volumes, in the realtime Beauty image,
//! against committed Blender 5.2.2 EEVEE captures.
//!
//! `fixtures/native_oracle/eevee_light_shapes_extra.py` saves three base scenes and three local-light cull variants;
//! `eevee_light_shape_boundaries.py` saves odd-viewport, scaled-spot, and area-cutoff scenes plus the four new boundary
//! cases:
//!
//! - a sun of angle 8° seen by a perspective camera that looks straight at it, whose disc EEVEE draws at
//!   `energy · (1 + 1 / r²) / π` with `r = tan(angle / 2)`;
//! - a spot light facing an orthographic camera, next to the area and point lights, whose disc EEVEE draws at the
//!   sphere emitter radiance `P / (4 π² r²)` on axis;
//! - the area light behind a slab of absorbing volume, with tile-size-8 and tile-size-1 captures; local cull variants
//!   verify that all-zero and transmission-only factors cull the shape while a volume-only factor keeps it visible;
//! - a hard point light whose physical sphere front is in front of an opaque plane, while EEVEE's center-plane emitter
//!   lies behind it, pinning the camera-shape depth test.
//! - orthographic and wide-FOV perspective point lights behind a mesh-volume absorption gradient that is zero at camera
//!   entry, pinning per-froxel material evaluation and jittered perspective prefix steps at the AA-filtered edge.
//! - a 101×65 perspective point shape through a heterogeneous mesh volume at tile size 16, plus a scaled off-axis spot,
//!   an orthographic off-centre on-axis spot, a perspective spot whose attenuation varies across its disc, a surviving
//!   0.002×2 rectangle, and a 0.004-square area light below EEVEE's half-extent cutoff.
//! - a BLENDED half-pane with a tinted Transparent BSDF on the left half of the frame: Blender draws the point shape
//!   above the pane's forward-transparency pass, so its radiance matches the pane-free ray. The test writes a real
//!   Combined+Depth multilayer EXR and checks raw channels at alpha 0, 0.5, and 1, plus a nonzero fractional-alpha
//!   base-only control, against Blender readback captures.
//!
//! The test imports each saved `.blend`, renders it with `pot render --engine realtime`, and compares against committed
//! macOS EEVEE pixel and transparent-film EXR readback captures. The sun, spots, and opaque-surface tolerances are 0.01;
//! the existing absorber tolerance remains 0.03 for tile sizes 8 and 1 and the gradient-volume center and edge.
//! Blender-gated: with `POTTER_REQUIRE_BLENDER` set, a missing Blender fails the test.

#[path = "common/blender_checked.rs"]
mod blender_checked;
#[path = "common/native_oracle.rs"]
#[expect(
    dead_code,
    reason = "each test binary uses a subset of the shared oracle helpers"
)]
mod native_oracle;
#[path = "common/process.rs"]
mod process;

use std::{
    error::Error,
    fs,
    path::{Path, PathBuf},
};

use native_oracle::{
    TestResult, apply_operations, import_blend, pot, read_scene, run_blender_arguments,
    write_script,
};
use potter_core::image::{ImageInterpolation, decode_pixels, load_image_data, sample};
use potter_core::model::{Image, ImageColorspace};
use serde_json::json;
use tempfile::tempdir;

/// The scenes and EEVEE renders (`eevee_light_shapes_extra.py`).
const SCRIPT: &str = include_str!("fixtures/native_oracle/eevee_light_shapes_extra.py");
const BOUNDARY_SCRIPT: &str =
    include_str!("fixtures/native_oracle/eevee_light_shape_boundaries.py");
const ODD_VIEWPORT_WIDTH: usize = 101;
const ODD_VIEWPORT_HEIGHT: usize = 65;
const ODD_VIEWPORT_EDGE_COLUMN: usize = 22;
const ODD_VIEWPORT_EDGE_ROW: usize = 32;
const EEVEE_MESH_VOLUME_ODD_PERSPECTIVE: &[u8] =
    include_bytes!("fixtures/native_oracle/eevee_macos/mesh_volume_gradient_odd_perspective.f32le");
const EEVEE_SPOT_SCALED_OFFAXIS: &[u8] =
    include_bytes!("fixtures/native_oracle/eevee_macos/spot_scaled_offaxis.f32le");
const EEVEE_AREA_SMALL_CUTOFF: &[u8] =
    include_bytes!("fixtures/native_oracle/eevee_macos/area_small_cutoff.f32le");
const EEVEE_SPOT_ORTHO_OFFCENTER: &[u8] =
    include_bytes!("fixtures/native_oracle/eevee_macos/spot_ortho_offcenter.f32le");
const EEVEE_SPOT_PERSPECTIVE_RAY: &[u8] =
    include_bytes!("fixtures/native_oracle/eevee_macos/spot_perspective_ray.f32le");
const EEVEE_AREA_SMALL_SURVIVOR: &[u8] =
    include_bytes!("fixtures/native_oracle/eevee_macos/area_small_survivor.f32le");
const EEVEE_AREA_INFLUENCE_CULLED: &[u8] =
    include_bytes!("fixtures/native_oracle/eevee_macos/area_influence_culled.f32le");
const BLEND_TRANSPARENCY_SCRIPT: &str =
    include_str!("fixtures/native_oracle/eevee_light_shape_transparency.py");
const EEVEE_BLEND_CLEAR_EXR: &[u8] =
    include_bytes!("fixtures/native_oracle/eevee_macos/light_shape_blend_clear.exr");
const EEVEE_BLEND_PANE_EXR: &[u8] =
    include_bytes!("fixtures/native_oracle/eevee_macos/light_shape_blend_pane.exr");
const FRACTIONAL_BASE_SCRIPT: &str =
    include_str!("fixtures/native_oracle/eevee_light_shape_fractional_base.py");
const EEVEE_BLEND_FRACTIONAL_BASE_EXR: &[u8] =
    include_bytes!("fixtures/native_oracle/eevee_macos/light_shape_blend_fractional_base.exr");

struct BoundaryScenes {
    odd_viewport: PathBuf,
    scaled_spot: PathBuf,
    small_area_cutoff: PathBuf,
    ortho_spot_offcenter: PathBuf,
    perspective_spot: PathBuf,
    small_area_survivor: PathBuf,
    area_influence_culled: PathBuf,
}

fn generate_boundary_scenes(blender: &Path, root: &Path) -> Result<BoundaryScenes, Box<dyn Error>> {
    let script = write_script(root, "eevee_light_shape_boundaries.py", BOUNDARY_SCRIPT)?;
    let scenes = BoundaryScenes {
        odd_viewport: root.join("odd_viewport.blend"),
        scaled_spot: root.join("scaled_spot.blend"),
        small_area_cutoff: root.join("small_area.blend"),
        ortho_spot_offcenter: root.join("ortho_spot_offcenter.blend"),
        perspective_spot: root.join("perspective_spot.blend"),
        small_area_survivor: root.join("small_area_survivor.blend"),
        area_influence_culled: root.join("area_influence_culled.blend"),
    };
    run_blender_arguments(
        blender,
        &script,
        None,
        &[
            scenes.odd_viewport.as_path(),
            scenes.scaled_spot.as_path(),
            scenes.small_area_cutoff.as_path(),
            scenes.ortho_spot_offcenter.as_path(),
            scenes.perspective_spot.as_path(),
            scenes.small_area_survivor.as_path(),
            scenes.area_influence_culled.as_path(),
            Path::new("-"),
            Path::new("-"),
            Path::new("-"),
            Path::new("-"),
            Path::new("-"),
            Path::new("-"),
            Path::new("-"),
        ],
    )?;
    Ok(scenes)
}

struct TransparencyScenes {
    clear: PathBuf,
    pane: PathBuf,
}

fn generate_transparency_scenes(
    blender: &Path,
    root: &Path,
) -> Result<TransparencyScenes, Box<dyn Error>> {
    let script = write_script(
        root,
        "eevee_light_shape_transparency.py",
        BLEND_TRANSPARENCY_SCRIPT,
    )?;
    let pane_source = root.join("blend_pane.blend");
    let scenes = TransparencyScenes {
        clear: root.join("blend_clear.blend"),
        pane: root.join("blend_pane_with_fractional_base.blend"),
    };
    let unused_clear = root.join("unused_clear.exr");
    let unused_pane = root.join("unused_pane.exr");
    run_blender_arguments(
        blender,
        &script,
        None,
        &[
            scenes.clear.as_path(),
            unused_clear.as_path(),
            pane_source.as_path(),
            unused_pane.as_path(),
            Path::new("0"),
        ],
    )?;
    let base_script = write_script(
        root,
        "eevee_light_shape_fractional_base.py",
        FRACTIONAL_BASE_SCRIPT,
    )?;
    let unused_base = root.join("unused_fractional_base.exr");
    run_blender_arguments(
        blender,
        &base_script,
        None,
        &[
            pane_source.as_path(),
            scenes.pane.as_path(),
            unused_base.as_path(),
            Path::new("0"),
        ],
    )?;
    Ok(scenes)
}

/// The mean red value of the `(2 * half + 1)` square window centred on `(column, row)` (top-down) in the image `id`.
fn window_mean(
    project: &Path,
    id: &str,
    column: usize,
    row: usize,
    half: usize,
) -> Result<f64, Box<dyn Error>> {
    window_mean_at(
        project,
        id,
        column,
        row,
        half,
        RESOLUTION as usize,
        RESOLUTION as usize,
    )
}

fn window_mean_at(
    project: &Path,
    id: &str,
    column: usize,
    row: usize,
    half: usize,
    width: usize,
    height: usize,
) -> Result<f64, Box<dyn Error>> {
    let scene = read_scene(project)?;
    let image: Image = serde_json::from_value(scene["images"][id].clone())?;
    let data = load_image_data(&image, project, ImageInterpolation::Closest)?;
    let width = width as f64;
    let height = height as f64;
    let mut total = 0.0;
    let mut count = 0.0;
    for r in row - half..=row + half {
        for c in column - half..=column + half {
            let uv = [(c as f64 + 0.5) / width, (r as f64 + 0.5) / height];
            total += sample(&data, uv, false)[0];
            count += 1.0;
        }
    }
    Ok(total / count)
}

/// The mean red value of an RGB f32le capture window; fixture rows are bottom-up, unlike Potter's top-down rows.
fn oracle_window_mean(capture: &[u8], column: usize, row: usize, half: usize) -> f64 {
    oracle_window_mean_at(
        capture,
        column,
        row,
        half,
        RESOLUTION as usize,
        RESOLUTION as usize,
    )
}

fn oracle_window_mean_at(
    capture: &[u8],
    column: usize,
    row: usize,
    half: usize,
    width: usize,
    height: usize,
) -> f64 {
    assert_eq!(capture.len(), width * height * 3 * 4);
    let mut total = 0.0;
    for top_down_row in row - half..=row + half {
        let bottom_up_row = height - 1 - top_down_row;
        for sample_column in column - half..=column + half {
            let offset = (bottom_up_row * width + sample_column) * 3 * 4;
            let red = f32::from_le_bytes([
                capture[offset],
                capture[offset + 1],
                capture[offset + 2],
                capture[offset + 3],
            ]);
            total += f64::from(red);
        }
    }
    total / ((2 * half + 1) * (2 * half + 1)) as f64
}

/// Imports `blend` into `root/name`, renders it in the realtime engine with `samples` samples, and loads the frame as
/// the image `id`. Returns the project's directory.
fn render_realtime(
    blender: &Path,
    root: &Path,
    name: &str,
    blend: &Path,
    id: &str,
    volumetric_tile_size: u32,
) -> Result<std::path::PathBuf, Box<dyn Error>> {
    render_realtime_at(
        blender,
        root,
        name,
        blend,
        id,
        RESOLUTION,
        RESOLUTION,
        volumetric_tile_size,
    )
}

fn render_realtime_at(
    blender: &Path,
    root: &Path,
    name: &str,
    blend: &Path,
    id: &str,
    width: u32,
    height: u32,
    volumetric_tile_size: u32,
) -> Result<std::path::PathBuf, Box<dyn Error>> {
    let (project, _) = import_blend(blender, root, name, blend)?;
    let scene = read_scene(&project)?;
    let active = scene["active_scene"]
        .as_str()
        .ok_or("active scene is missing")?
        .to_owned();
    let applied = apply_operations(
        &project,
        root,
        json!([
            {"op": "render.update", "target": {"id": active}, "set": {"resolution_x": width, "resolution_y": height, "samples": 64, "seed": 0, "eevee_volumetric_tile_size": volumetric_tile_size}}
        ]),
    )?;
    assert_eq!(applied["ok"], true, "{applied}");
    let frames = root.join(format!("frames_{name}"));
    let rendered = pot()
        .arg("render")
        .arg(&project)
        .args(["--engine", "realtime", "--format", "exr", "--out"])
        .arg(&frames)
        .arg("--json")
        .output()?;
    assert!(
        rendered.status.success(),
        "realtime render failed: {}",
        String::from_utf8_lossy(&rendered.stdout)
    );
    let loaded = apply_operations(
        &project,
        root,
        json!([{"op": "image.load", "id": id, "path": frames.join("frame_0001.exr").to_string_lossy()}]),
    )?;
    assert_eq!(loaded["ok"], true, "{loaded}");
    Ok(project)
}

fn decode_exr_pixels(bytes: &[u8], label: &str) -> TestResult<Vec<[f64; 4]>> {
    assert_eq!(
        bytes.get(..4),
        Some(&[0x76, 0x2f, 0x31, 0x01][..]),
        "{label} is not an OpenEXR file"
    );
    let (width, height, pixels) = decode_pixels(bytes, ImageColorspace::Linear)?;
    assert_eq!((width, height), (RESOLUTION, RESOLUTION));
    Ok(pixels)
}

fn decode_multilayer_exr(bytes: &[u8], label: &str) -> TestResult<Vec<[f64; 4]>> {
    for layer in ["Combined", "Depth"] {
        assert!(
            bytes
                .windows(layer.len())
                .any(|window| window == layer.as_bytes()),
            "{label} EXR is missing its `{layer}` layer",
        );
    }
    decode_exr_pixels(bytes, label)
}

fn render_multilayer_realtime(
    blender: &Path,
    root: &Path,
    name: &str,
    blend: &Path,
) -> TestResult<Vec<[f64; 4]>> {
    let (project, _) = import_blend(blender, root, name, blend)?;
    let scene = read_scene(&project)?;
    let active = scene["active_scene"]
        .as_str()
        .ok_or("active scene is missing")?
        .to_owned();
    let applied = apply_operations(
        &project,
        root,
        json!([
            {"op": "render.update", "target": {"id": active}, "set": {"resolution_x": RESOLUTION, "resolution_y": RESOLUTION, "samples": 64, "seed": 0, "film_transparent": true}},
            {"op": "render.passes_update", "target": {"id": active}, "set": {"passes": ["combined", "z"]}}
        ]),
    )?;
    assert_eq!(applied["ok"], true, "{applied}");
    let frames = root.join(format!("frames_{name}"));
    let rendered = pot()
        .arg("render")
        .arg(&project)
        .args(["--engine", "realtime", "--format", "exr", "--out"])
        .arg(&frames)
        .arg("--json")
        .output()?;
    assert!(
        rendered.status.success(),
        "realtime multilayer render failed: {}",
        String::from_utf8_lossy(&rendered.stdout)
    );
    let bytes = fs::read(frames.join("frame_0001.exr"))?;
    decode_multilayer_exr(&bytes, name)
}

/// Render settings shared with `eevee_light_shapes_extra.py`.
const RESOLUTION: u32 = 64;
const WINDOW: usize = 2;

/// Blender's bottom-up RGB f32le captures for the sun, spot, and absorber scenes.
const EEVEE_SUN: &[u8] = include_bytes!("fixtures/native_oracle/eevee_macos/sun.f32le");
const EEVEE_SPOT: &[u8] = include_bytes!("fixtures/native_oracle/eevee_macos/spot.f32le");
const EEVEE_ABSORBER: &[u8] = include_bytes!("fixtures/native_oracle/eevee_macos/absorber.f32le");
const EEVEE_ABSORBER_TILE1: &[u8] =
    include_bytes!("fixtures/native_oracle/eevee_macos/absorber_tile1.f32le");

const EEVEE_ABSORBER_ZERO: &[u8] =
    include_bytes!("fixtures/native_oracle/eevee_macos/absorber_zero.f32le");
const EEVEE_ABSORBER_VOLUME_ONLY: &[u8] =
    include_bytes!("fixtures/native_oracle/eevee_macos/absorber_volume_only.f32le");
const EEVEE_ABSORBER_TRANSMISSION_ONLY: &[u8] =
    include_bytes!("fixtures/native_oracle/eevee_macos/absorber_transmission_only.f32le");

const OCCLUSION_SCRIPT: &str =
    include_str!("fixtures/native_oracle/eevee_light_shape_occlusion.py");
const EEVEE_POINT_OCCLUDED: &[u8] =
    include_bytes!("fixtures/native_oracle/eevee_macos/point_occluded.f32le");
const MESH_VOLUME_SCRIPT: &str =
    include_str!("fixtures/native_oracle/eevee_mesh_volume_gradient.py");
const EEVEE_MESH_VOLUME_GRADIENT: &[u8] =
    include_bytes!("fixtures/native_oracle/eevee_macos/mesh_volume_gradient.f32le");
const EEVEE_MESH_VOLUME_PERSPECTIVE: &[u8] =
    include_bytes!("fixtures/native_oracle/eevee_macos/mesh_volume_gradient_perspective.f32le");
const MESH_VOLUME_PERSPECTIVE_COLUMN: usize = 11;

/// Linear tolerance on a window mean against EEVEE, for the sun and the spot.
const EMITTER_TOLERANCE: f64 = 0.01;

/// The existing 0.03 linear window-mean tolerance for absorber and heterogeneous mesh-volume EEVEE comparisons.
const ABSORBER_TOLERANCE: f64 = 0.03;

#[test]
fn realtime_sun_spot_and_absorber_match_eevee() -> TestResult {
    let Some(blender) = blender_checked::blender_executable() else {
        return Ok(());
    };
    let root = tempdir()?;
    let script = write_script(root.path(), "eevee_light_shapes_extra.py", SCRIPT)?;
    let sun_blend = root.path().join("sun.blend");
    let spot_blend = root.path().join("spot.blend");
    let absorber_blend = root.path().join("absorber.blend");
    let absorber_zero_blend = root.path().join("absorber_zero.blend");
    let absorber_volume_blend = root.path().join("absorber_volume.blend");
    let absorber_transmission_blend = root.path().join("absorber_transmission.blend");
    run_blender_arguments(
        &blender,
        &script,
        None,
        &[
            sun_blend.as_path(),
            spot_blend.as_path(),
            absorber_blend.as_path(),
            absorber_zero_blend.as_path(),
            absorber_volume_blend.as_path(),
            absorber_transmission_blend.as_path(),
            Path::new("-"),
            Path::new("-"),
            Path::new("-"),
            Path::new("-"),
            Path::new("-"),
            Path::new("-"),
            Path::new("-"),
            Path::new("-"),
        ],
    )?;
    let centre = RESOLUTION as usize / 2;

    // The sun: the centre of the disc and a corner outside it.
    let project = render_realtime(&blender, root.path(), "sun", &sun_blend, "sun", 8)?;
    let sun_centre = window_mean(&project, "sun", centre, centre, WINDOW)?;
    let sun_corner = window_mean(&project, "sun", 8, 8, WINDOW)?;
    let expected_sun_centre = oracle_window_mean(EEVEE_SUN, centre, centre, WINDOW);
    let expected_sun_corner = oracle_window_mean(EEVEE_SUN, 8, 8, WINDOW);
    assert!(
        (sun_centre - expected_sun_centre).abs() <= EMITTER_TOLERANCE,
        "sun disc: Potter {sun_centre} differs from EEVEE {expected_sun_centre}"
    );
    assert!(
        (sun_corner - expected_sun_corner).abs() <= EMITTER_TOLERANCE / 2.0,
        "sun corner: Potter {sun_corner} differs from EEVEE {expected_sun_corner}"
    );

    // The spot emitter on axis and the area light beside it.
    let project = render_realtime(&blender, root.path(), "spot", &spot_blend, "spot", 8)?;
    let spot = window_mean(&project, "spot", 32, 12, WINDOW)?;
    let expected_spot = oracle_window_mean(EEVEE_SPOT, 32, 12, WINDOW);
    assert!(
        (spot - expected_spot).abs() <= EMITTER_TOLERANCE,
        "spot emitter: Potter {spot} differs from EEVEE {expected_spot}"
    );
    let area_clear = window_mean(&project, "spot", 20, centre, WINDOW)?;
    let expected_area_clear = oracle_window_mean(EEVEE_SPOT, 20, centre, WINDOW);
    assert!(
        (area_clear - expected_area_clear).abs() <= EMITTER_TOLERANCE,
        "area light (no absorber): Potter {area_clear} differs from EEVEE {expected_area_clear}"
    );

    // The area light seen through the absorber: attenuated, and within the froxel bound of EEVEE.
    let project = render_realtime(
        &blender,
        root.path(),
        "absorber",
        &absorber_blend,
        "absorber",
        8,
    )?;
    let through = window_mean(&project, "absorber", 20, centre, WINDOW)?;
    let expected_through = oracle_window_mean(EEVEE_ABSORBER, 20, centre, WINDOW);
    assert!(
        through < 0.9 * expected_area_clear,
        "the absorber must darken the area light, got {through}"
    );
    assert!(
        (through - expected_through).abs() <= ABSORBER_TOLERANCE,
        "area light through the absorber: Potter {through} differs from EEVEE {expected_through}"
    );

    let project = render_realtime(
        &blender,
        root.path(),
        "absorber_tile1",
        &absorber_blend,
        "absorber_tile1",
        1,
    )?;
    let through_tile1 = window_mean(&project, "absorber_tile1", 20, centre, WINDOW)?;
    let expected_through_tile1 = oracle_window_mean(EEVEE_ABSORBER_TILE1, 20, centre, WINDOW);
    assert!(
        through_tile1 < 0.9 * expected_area_clear,
        "the absorber must darken the tile-1 area light, got {through_tile1}"
    );
    assert!(
        (through_tile1 - expected_through_tile1).abs() <= ABSORBER_TOLERANCE,
        "area light through the absorber at tile size 1: Potter {through_tile1} differs from EEVEE {expected_through_tile1}"
    );
    // EEVEE's local-light cull list uses diffuse/specular and volume power, not transmission.
    for (name, blend, capture, tolerance) in [
        (
            "absorber_zero",
            &absorber_zero_blend,
            EEVEE_ABSORBER_ZERO,
            EMITTER_TOLERANCE,
        ),
        (
            "absorber_volume",
            &absorber_volume_blend,
            EEVEE_ABSORBER_VOLUME_ONLY,
            ABSORBER_TOLERANCE,
        ),
        (
            "absorber_transmission",
            &absorber_transmission_blend,
            EEVEE_ABSORBER_TRANSMISSION_ONLY,
            EMITTER_TOLERANCE,
        ),
    ] {
        let project = render_realtime(&blender, root.path(), name, blend, name, 8)?;
        let actual = window_mean(&project, name, 20, centre, WINDOW)?;
        let expected = oracle_window_mean(capture, 20, centre, WINDOW);
        assert!(
            (actual - expected).abs() <= tolerance,
            "{name}: Potter {actual} differs from EEVEE {expected}"
        );
        if name == "absorber_volume" {
            assert!(
                (actual - through).abs() <= EMITTER_TOLERANCE,
                "camera-shape radiance changed when only the volume cull factor remained: {actual} vs {through}"
            );
        }
    }
    Ok(())
}

#[test]
fn realtime_point_shape_uses_center_plane_depth_for_surface_occlusion() -> TestResult {
    let Some(blender) = blender_checked::blender_executable() else {
        return Ok(());
    };
    let root = tempdir()?;
    let script = write_script(
        root.path(),
        "eevee_light_shape_occlusion.py",
        OCCLUSION_SCRIPT,
    )?;
    let blend = root.path().join("point_occluded.blend");
    run_blender_arguments(&blender, &script, None, &[blend.as_path(), Path::new("-")])?;

    let project = render_realtime(
        &blender,
        root.path(),
        "point_occluded",
        &blend,
        "point_occluded",
        8,
    )?;
    let centre = RESOLUTION as usize / 2;
    let actual = window_mean(&project, "point_occluded", centre, centre, WINDOW)?;
    let expected = oracle_window_mean(EEVEE_POINT_OCCLUDED, centre, centre, WINDOW);
    assert!(
        expected.abs() <= 1.0e-6,
        "Blender's opaque depth oracle should contain no point-shape radiance, found {expected}"
    );
    assert!(
        (actual - expected).abs() <= EMITTER_TOLERANCE,
        "camera-visible point shape at the light center plane: Potter {actual} differs from EEVEE {expected}"
    );
    Ok(())
}

#[test]
fn realtime_blended_pane_preserves_camera_shape_radiance_in_multilayer_exr() -> TestResult {
    let Some(blender) = blender_checked::blender_executable() else {
        return Ok(());
    };
    let root = tempdir()?;
    let scenes = generate_transparency_scenes(&blender, root.path())?;
    let native_clear = decode_exr_pixels(EEVEE_BLEND_CLEAR_EXR, "EEVEE pane-free")?;
    let native_pane = decode_exr_pixels(EEVEE_BLEND_PANE_EXR, "EEVEE BLENDED pane")?;
    let native_fractional_base = decode_exr_pixels(
        EEVEE_BLEND_FRACTIONAL_BASE_EXR,
        "EEVEE fractional-alpha base-only",
    )?;
    let clear = render_multilayer_realtime(&blender, root.path(), "blend_clear", &scenes.clear)?;
    let pane = render_multilayer_realtime(&blender, root.path(), "blend_pane", &scenes.pane)?;

    let pixel = |pixels: &[[f64; 4]], x: usize, y: usize| pixels[y * RESOLUTION as usize + x];
    let clear_alpha_zero_site = pixel(&native_clear, 30, 32);
    let native_fractional_site = pixel(&native_pane, 30, 32);
    let native_alpha_zero_site = pixel(&native_pane, 33, 32);
    let clear_shape_at_opaque_site = pixel(&native_clear, 25, 32);
    let native_shape_at_opaque_site = pixel(&native_pane, 25, 32);
    let native_opaque_base_site = pixel(&native_pane, 52, 32);
    let native_fractional_base_site = pixel(&native_fractional_base, 52, 12);
    let native_control_frame_shape_site = pixel(&native_fractional_base, 30, 32);
    let potter_clear_site = pixel(&clear, 30, 32);
    let potter_fractional_site = pixel(&pane, 30, 32);
    let potter_alpha_zero_site = pixel(&pane, 33, 32);
    let potter_shape_at_opaque_site = pixel(&pane, 25, 32);
    let potter_opaque_base_site = pixel(&pane, 52, 32);
    let potter_fractional_base_site = pixel(&pane, 52, 12);

    assert!(clear_alpha_zero_site[3].abs() <= 1.0e-6);
    assert!(
        (native_fractional_site[3] - 0.5).abs() <= EMITTER_TOLERANCE,
        "native pane sample should have alpha 0.5: {}",
        native_fractional_site[3]
    );
    assert!(native_alpha_zero_site[3].abs() <= 1.0e-6);
    assert!(
        (native_shape_at_opaque_site[3] - 1.0).abs() <= 1.0e-6,
        "native opaque backing sample should have alpha 1: {}",
        native_shape_at_opaque_site[3]
    );
    assert!(
        (native_opaque_base_site[3] - 1.0).abs() <= 1.0e-6,
        "native opaque base sample should have alpha 1: {}",
        native_opaque_base_site[3]
    );
    for (channel, expected) in [0.25, 0.1, 0.05].into_iter().enumerate() {
        assert!(
            (native_opaque_base_site[channel] - expected).abs() <= 1.0e-3,
            "native opaque base channel {channel} is {}, expected {expected}",
            native_opaque_base_site[channel]
        );
    }
    assert!(
        (native_fractional_base_site[3] - 0.5).abs() <= EMITTER_TOLERANCE,
        "native no-shape control should have alpha 0.5: {}",
        native_fractional_base_site[3]
    );
    for (channel, expected) in [0.125, 0.05, 0.025].into_iter().enumerate() {
        assert!(
            (native_fractional_base_site[channel] - expected).abs() <= 1.0e-3,
            "native no-shape base channel {channel} is {}, expected {expected}",
            native_fractional_base_site[channel]
        );
    }
    assert!(
        (native_control_frame_shape_site[3] - 0.5).abs() <= EMITTER_TOLERANCE,
        "fractional-base control frame should retain its pane alpha at the emitter: {}",
        native_control_frame_shape_site[3]
    );
    for channel in 0..3 {
        assert!(
            native_control_frame_shape_site[channel] > 0.01
                && (native_control_frame_shape_site[channel] - native_fractional_site[channel])
                    .abs()
                    <= EMITTER_TOLERANCE,
            "fractional-base control frame lost the camera-visible shape on channel {channel}"
        );
    }
    for channel in 0..3 {
        for (label, sample) in [
            ("fractional-alpha pane", native_fractional_site),
            ("alpha-zero pane", native_alpha_zero_site),
            ("opaque backing", native_shape_at_opaque_site),
        ] {
            assert!(
                sample[channel] > 0.01,
                "native EXR lost {label} camera-shape radiance on channel {channel}"
            );
        }
        assert!(
            (native_fractional_site[channel] - clear_alpha_zero_site[channel]).abs()
                <= EMITTER_TOLERANCE,
            "Blender tinted the light shape through its BLENDED pane on channel {channel}"
        );
        assert!(
            (native_shape_at_opaque_site[channel] - clear_shape_at_opaque_site[channel]).abs()
                <= EMITTER_TOLERANCE,
            "opaque backing changed additive camera-shape radiance on channel {channel}"
        );
    }
    for (label, expected, actual) in [
        (
            "pane-free alpha zero",
            clear_alpha_zero_site,
            potter_clear_site,
        ),
        (
            "BLENDED pane alpha 0.5",
            native_fractional_site,
            potter_fractional_site,
        ),
        (
            "BLENDED pane alpha zero",
            native_alpha_zero_site,
            potter_alpha_zero_site,
        ),
        (
            "opaque alpha-one light shape",
            native_shape_at_opaque_site,
            potter_shape_at_opaque_site,
        ),
        (
            "opaque alpha-one base",
            native_opaque_base_site,
            potter_opaque_base_site,
        ),
        (
            "fractional-alpha base-only",
            native_fractional_base_site,
            potter_fractional_base_site,
        ),
    ] {
        for channel in 0..4 {
            assert!(
                (actual[channel] - expected[channel]).abs() <= EMITTER_TOLERANCE,
                "{label} raw Combined channel {channel}: Potter stored {}, EEVEE stored {}",
                actual[channel],
                expected[channel]
            );
        }
    }
    assert!(
        (potter_fractional_site[3] - 0.5).abs() <= EMITTER_TOLERANCE,
        "Potter pane sample should retain alpha 0.5: {}",
        potter_fractional_site[3]
    );
    assert!(
        potter_alpha_zero_site[3].abs() <= 1.0e-6,
        "Potter alpha-zero sample should remain transparent: {}",
        potter_alpha_zero_site[3]
    );
    assert!(
        (potter_fractional_base_site[3] - 0.5).abs() <= EMITTER_TOLERANCE,
        "Potter no-shape control should retain alpha 0.5: {}",
        potter_fractional_base_site[3]
    );
    assert!(
        (potter_shape_at_opaque_site[3] - 1.0).abs() <= 1.0e-6
            && (potter_opaque_base_site[3] - 1.0).abs() <= 1.0e-6,
        "Potter opaque samples should retain alpha 1: shape {}, base {}",
        potter_shape_at_opaque_site[3],
        potter_opaque_base_site[3]
    );
    Ok(())
}

#[test]
fn realtime_point_shape_evaluates_heterogeneous_mesh_volume_per_froxel_sample() -> TestResult {
    let Some(blender) = blender_checked::blender_executable() else {
        return Ok(());
    };
    let root = tempdir()?;
    let script = write_script(
        root.path(),
        "eevee_mesh_volume_gradient.py",
        MESH_VOLUME_SCRIPT,
    )?;
    let blend = root.path().join("mesh_volume_gradient.blend");
    let capture = root.path().join("mesh_volume_gradient.f32le");
    run_blender_arguments(
        &blender,
        &script,
        None,
        &[blend.as_path(), capture.as_path()],
    )?;

    let project = render_realtime(
        &blender,
        root.path(),
        "mesh_volume_gradient",
        &blend,
        "mesh_volume_gradient",
        2,
    )?;
    let centre = RESOLUTION as usize / 2;
    let actual_centre = window_mean(&project, "mesh_volume_gradient", centre, centre, WINDOW)?;
    let expected_centre = oracle_window_mean(EEVEE_MESH_VOLUME_GRADIENT, centre, centre, WINDOW);
    assert!(
        (actual_centre - expected_centre).abs() <= ABSORBER_TOLERANCE,
        "heterogeneous mesh-volume attenuation at the point-shape center: Potter {actual_centre} differs from EEVEE {expected_centre}"
    );

    let edge_column = centre + 7;
    let actual_edge = window_mean(&project, "mesh_volume_gradient", edge_column, centre, 0)?;
    let expected_edge = oracle_window_mean(EEVEE_MESH_VOLUME_GRADIENT, edge_column, centre, 0);
    assert!(
        (actual_edge - expected_edge).abs() <= ABSORBER_TOLERANCE,
        "AA-filtered point-shape edge through heterogeneous volume: Potter {actual_edge} differs from EEVEE {expected_edge}"
    );
    Ok(())
}

#[test]
fn realtime_perspective_mesh_volume_uses_the_sample_jittered_column_ray() -> TestResult {
    let Some(blender) = blender_checked::blender_executable() else {
        return Ok(());
    };
    let root = tempdir()?;
    let script = write_script(
        root.path(),
        "eevee_mesh_volume_gradient.py",
        MESH_VOLUME_SCRIPT,
    )?;
    let blend = root.path().join("mesh_volume_gradient_perspective.blend");
    let capture = root.path().join("mesh_volume_gradient_perspective.f32le");
    run_blender_arguments(
        &blender,
        &script,
        None,
        &[
            blend.as_path(),
            capture.as_path(),
            Path::new("PERSP"),
            Path::new("16"),
            Path::new("-6.0"),
        ],
    )?;

    let project = render_realtime(
        &blender,
        root.path(),
        "mesh_volume_gradient_perspective",
        &blend,
        "mesh_volume_gradient_perspective",
        16,
    )?;
    let actual_center = window_mean(
        &project,
        "mesh_volume_gradient_perspective",
        MESH_VOLUME_PERSPECTIVE_COLUMN,
        RESOLUTION as usize / 2,
        WINDOW,
    )?;
    let expected_center = oracle_window_mean(
        EEVEE_MESH_VOLUME_PERSPECTIVE,
        MESH_VOLUME_PERSPECTIVE_COLUMN,
        RESOLUTION as usize / 2,
        WINDOW,
    );
    assert!(
        (actual_center - expected_center).abs() <= ABSORBER_TOLERANCE,
        "perspective froxel attenuation at the point-shape center: Potter {actual_center} differs from EEVEE {expected_center}"
    );

    let edge_column = MESH_VOLUME_PERSPECTIVE_COLUMN + 3;
    let actual_edge = window_mean(
        &project,
        "mesh_volume_gradient_perspective",
        edge_column,
        RESOLUTION as usize / 2,
        0,
    )?;
    let expected_edge = oracle_window_mean(
        EEVEE_MESH_VOLUME_PERSPECTIVE,
        edge_column,
        RESOLUTION as usize / 2,
        0,
    );
    assert!(
        (actual_edge - expected_edge).abs() <= ABSORBER_TOLERANCE,
        "wide-FOV AA edge through coarse perspective froxels: Potter {actual_edge} differs from EEVEE {expected_edge}"
    );
    Ok(())
}

#[test]
fn realtime_light_shape_boundaries_match_eevee() -> TestResult {
    let Some(blender) = blender_checked::blender_executable() else {
        return Ok(());
    };
    let root = tempdir()?;
    let scenes = generate_boundary_scenes(&blender, root.path())?;

    let project = render_realtime_at(
        &blender,
        root.path(),
        "odd_viewport",
        &scenes.odd_viewport,
        "odd_viewport",
        ODD_VIEWPORT_WIDTH as u32,
        ODD_VIEWPORT_HEIGHT as u32,
        16,
    )?;
    let odd_edge = window_mean_at(
        &project,
        "odd_viewport",
        ODD_VIEWPORT_EDGE_COLUMN,
        ODD_VIEWPORT_EDGE_ROW,
        0,
        ODD_VIEWPORT_WIDTH,
        ODD_VIEWPORT_HEIGHT,
    )?;
    let expected_odd_edge = oracle_window_mean_at(
        EEVEE_MESH_VOLUME_ODD_PERSPECTIVE,
        ODD_VIEWPORT_EDGE_COLUMN,
        ODD_VIEWPORT_EDGE_ROW,
        0,
        ODD_VIEWPORT_WIDTH,
        ODD_VIEWPORT_HEIGHT,
    );
    assert!(
        expected_odd_edge > 0.0,
        "the native odd-viewport volume edge must retain the visible point shape"
    );
    assert!(
        (odd_edge - expected_odd_edge).abs() <= ABSORBER_TOLERANCE,
        "odd-extent perspective AA edge through heterogeneous volume: Potter {odd_edge} differs from EEVEE {expected_odd_edge}"
    );

    let project = render_realtime(
        &blender,
        root.path(),
        "scaled_spot",
        &scenes.scaled_spot,
        "scaled_spot",
        8,
    )?;
    let spot_center = window_mean(&project, "scaled_spot", 32, 32, 0)?;
    let expected_spot_center = oracle_window_mean(EEVEE_SPOT_SCALED_OFFAXIS, 32, 32, 0);
    assert_eq!(
        expected_spot_center, 0.0,
        "the native scaled spot is outside its cone at the selected camera"
    );
    assert!(
        (spot_center - expected_spot_center).abs() <= EMITTER_TOLERANCE,
        "scaled off-axis spot: Potter {spot_center} differs from EEVEE {expected_spot_center}"
    );

    let project = render_realtime(
        &blender,
        root.path(),
        "small_area",
        &scenes.small_area_cutoff,
        "small_area",
        8,
    )?;
    let area_center = window_mean(&project, "small_area", 32, 32, 0)?;
    let expected_area_center = oracle_window_mean(EEVEE_AREA_SMALL_CUTOFF, 32, 32, 0);
    assert_eq!(
        expected_area_center, 0.0,
        "the native area light is culled below EEVEE's half-extent product cutoff"
    );
    assert!(
        (area_center - expected_area_center).abs() <= EMITTER_TOLERANCE,
        "small square area-light cutoff: Potter {area_center} differs from EEVEE {expected_area_center}"
    );
    Ok(())
}

#[test]
fn realtime_spot_shapes_match_native_camera_ray_attenuation() -> TestResult {
    let Some(blender) = blender_checked::blender_executable() else {
        return Ok(());
    };
    let root = tempdir()?;
    let scenes = generate_boundary_scenes(&blender, root.path())?;

    let project = render_realtime(
        &blender,
        root.path(),
        "ortho_spot_offcenter",
        &scenes.ortho_spot_offcenter,
        "ortho_spot_offcenter",
        8,
    )?;
    let ortho_spot = window_mean(&project, "ortho_spot_offcenter", 56, 32, 0)?;
    let expected_ortho_spot = oracle_window_mean(EEVEE_SPOT_ORTHO_OFFCENTER, 56, 32, 0);
    assert!(
        expected_ortho_spot > 0.0,
        "the native off-centre orthographic spot must retain its on-axis radiance"
    );
    assert!(
        (ortho_spot - expected_ortho_spot).abs() <= EMITTER_TOLERANCE,
        "off-centre orthographic spot: Potter {ortho_spot} differs from EEVEE {expected_ortho_spot}"
    );

    let project = render_realtime(
        &blender,
        root.path(),
        "perspective_spot",
        &scenes.perspective_spot,
        "perspective_spot",
        8,
    )?;
    let expected_left = oracle_window_mean(EEVEE_SPOT_PERSPECTIVE_RAY, 29, 32, 0);
    let expected_right = oracle_window_mean(EEVEE_SPOT_PERSPECTIVE_RAY, 35, 32, 0);
    assert!(
        expected_left > 0.0 && expected_right == 0.0,
        "the native perspective spot must attenuate across its disc: left {expected_left}, right {expected_right}"
    );
    for (column, expected) in [(29, expected_left), (35, expected_right)] {
        let actual = window_mean(&project, "perspective_spot", column, 32, 0)?;
        assert!(
            (actual - expected).abs() <= EMITTER_TOLERANCE,
            "perspective spot at column {column}: Potter {actual} differs from EEVEE {expected}"
        );
    }
    Ok(())
}

#[test]
fn realtime_area_shapes_match_native_size_and_influence_culls() -> TestResult {
    let Some(blender) = blender_checked::blender_executable() else {
        return Ok(());
    };
    let root = tempdir()?;
    let scenes = generate_boundary_scenes(&blender, root.path())?;

    let project = render_realtime(
        &blender,
        root.path(),
        "small_area_survivor",
        &scenes.small_area_survivor,
        "small_area_survivor",
        8,
    )?;
    let survivor = window_mean(&project, "small_area_survivor", 32, 32, 0)?;
    let expected_survivor = oracle_window_mean(EEVEE_AREA_SMALL_SURVIVOR, 32, 32, 0);
    assert!(
        expected_survivor > 0.0,
        "the native 0.002 x 2 rectangle survives the half-extent cutoff and display clamp"
    );
    assert!(
        (survivor - expected_survivor).abs() <= EMITTER_TOLERANCE,
        "small surviving rectangle: Potter {survivor} differs from EEVEE {expected_survivor}"
    );

    let project = render_realtime(
        &blender,
        root.path(),
        "area_influence_culled",
        &scenes.area_influence_culled,
        "area_influence_culled",
        8,
    )?;
    let culled_edge = window_mean(&project, "area_influence_culled", 60, 32, 0)?;
    let expected_culled_edge = oracle_window_mean(EEVEE_AREA_INFLUENCE_CULLED, 60, 32, 0);
    assert_eq!(
        expected_culled_edge, 0.0,
        "EEVEE culls the area light's influence sphere even where its rectangle overlaps the view"
    );
    assert!(
        (culled_edge - expected_culled_edge).abs() <= EMITTER_TOLERANCE,
        "influence-culled area rectangle at the in-frame edge: Potter {culled_edge} differs from EEVEE {expected_culled_edge}"
    );
    Ok(())
}
