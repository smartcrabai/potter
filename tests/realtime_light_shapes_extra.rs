//! Sun, point, and spot emitters, and an area light through absorbing voxel and mesh volumes, in the realtime Beauty image,
//! against committed Blender 5.2.2 EEVEE captures.
//!
//! `fixtures/native_oracle/eevee_light_shapes_extra.py` saves three base scenes and three local-light cull variants;
//! `eevee_light_shape_occlusion.py` and `eevee_mesh_volume_gradient.py` save the opaque-surface and gradient-volume cases:
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
//!
//! The test imports each saved `.blend`, renders it with `pot render --engine realtime`, and compares window means with
//! committed macOS EEVEE pixel captures. The sun, spot, and opaque-surface tolerances are 0.01; the existing absorber
//! tolerance remains 0.03 for tile sizes 8 and 1 and the gradient-volume center and edge, with no tolerance change.
//!
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

use std::{error::Error, path::Path};

use native_oracle::{
    TestResult, apply_operations, import_blend, pot, read_scene, run_blender_arguments,
    write_script,
};
use potter_core::image::{ImageInterpolation, load_image_data, sample};
use potter_core::model::Image;
use serde_json::json;
use tempfile::tempdir;

/// The scenes and EEVEE renders (`eevee_light_shapes_extra.py`).
const SCRIPT: &str = include_str!("fixtures/native_oracle/eevee_light_shapes_extra.py");

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
const MESH_VOLUME_PERSPECTIVE_COLUMN: usize = 18;

/// Linear tolerance on a window mean against EEVEE, for the sun and the spot.
const EMITTER_TOLERANCE: f64 = 0.01;

/// The existing 0.03 linear window-mean tolerance for absorber and heterogeneous mesh-volume EEVEE comparisons.
const ABSORBER_TOLERANCE: f64 = 0.03;

/// The mean red value of the `(2 * half + 1)` square window centred on `(column, row)` (top-down) in the image `id`.
fn window_mean(
    project: &Path,
    id: &str,
    column: usize,
    row: usize,
    half: usize,
) -> Result<f64, Box<dyn Error>> {
    let scene = read_scene(project)?;
    let image: Image = serde_json::from_value(scene["images"][id].clone())?;
    let data = load_image_data(&image, project, ImageInterpolation::Closest)?;
    let texels = f64::from(RESOLUTION);
    let mut total = 0.0;
    let mut count = 0.0;
    for r in row - half..=row + half {
        for c in column - half..=column + half {
            let uv = [(c as f64 + 0.5) / texels, (r as f64 + 0.5) / texels];
            total += sample(&data, uv, false)[0];
            count += 1.0;
        }
    }
    Ok(total / count)
}

/// The mean red value of an RGB f32le capture window; fixture rows are bottom-up, unlike Potter's top-down rows.
fn oracle_window_mean(capture: &[u8], column: usize, row: usize, half: usize) -> f64 {
    let resolution = RESOLUTION as usize;
    assert_eq!(capture.len(), resolution * resolution * 3 * 4);
    let mut total = 0.0;
    for top_down_row in row - half..=row + half {
        let bottom_up_row = resolution - 1 - top_down_row;
        for sample_column in column - half..=column + half {
            let offset = (bottom_up_row * resolution + sample_column) * 3 * 4;
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
            {"op": "render.update", "target": {"id": active}, "set": {"resolution_x": RESOLUTION, "resolution_y": RESOLUTION, "samples": 64, "seed": 0, "eevee_volumetric_tile_size": volumetric_tile_size}}
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
        (actual - expected).abs() <= EMITTER_TOLERANCE,
        "camera-visible point shape at the light center plane: Potter {actual} differs from EEVEE {expected}"
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
            Path::new("-4.0"),
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
