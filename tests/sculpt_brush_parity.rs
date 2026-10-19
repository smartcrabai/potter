//! Blender 5.2.2 parity for the sculpt brushes: the dab-sequential brushes (`geom/sculpt_dabs.rs`
//! and its family modules) and the Pose and Boundary brushes (`geom/sculpt_dabs/`).
//!
//! The fixtures come from `tests/fixtures/sculpt_brushes/generate_oracle.py`, which drives Blender
//! in a session with a display (the script's header explains why). Each fixture holds the base mesh,
//! the stroke's dabs, and Blender's resulting vertex positions. Blender computes in single precision.
//! The tolerance of 5e-6 is set from that float32 noise, not fitted to Potter's output.

use std::{error::Error, fs, path::Path, process::Command};

use glam::DVec3;
use serde_json::{Value, json};

use potter_core::geom::{
    Mesh,
    sculpt::{self, BoundaryDeform, DabBrush, Falloff, PoseDeform, Stroke, StrokeSample},
};

const TOLERANCE: f64 = 5e-6;

const FIXTURES: &[(&str, &str)] = &[
    (
        "inflate_sphere",
        include_str!("fixtures/sculpt_brushes/inflate_sphere.json"),
    ),
    (
        "inflate_sphere_strength",
        include_str!("fixtures/sculpt_brushes/inflate_sphere_strength.json"),
    ),
    (
        "inflate_sphere_mirror",
        include_str!("fixtures/sculpt_brushes/inflate_sphere_mirror.json"),
    ),
    (
        "layer_sphere",
        include_str!("fixtures/sculpt_brushes/layer_sphere.json"),
    ),
    (
        "layer_sphere_clamp",
        include_str!("fixtures/sculpt_brushes/layer_sphere_clamp.json"),
    ),
    (
        "layer_sphere_mirror",
        include_str!("fixtures/sculpt_brushes/layer_sphere_mirror.json"),
    ),
    (
        "layer_sphere_masked",
        include_str!("fixtures/sculpt_brushes/layer_sphere_masked.json"),
    ),
    (
        "clay_strips_grid",
        include_str!("fixtures/sculpt_brushes/clay_strips_grid.json"),
    ),
    (
        "clay_strips_grid_mirror",
        include_str!("fixtures/sculpt_brushes/clay_strips_grid_mirror.json"),
    ),
    (
        "clay_strips_grid_pressure",
        include_str!("fixtures/sculpt_brushes/clay_strips_grid_pressure.json"),
    ),
    (
        "draw_sphere",
        include_str!("fixtures/sculpt_brushes/draw_sphere.json"),
    ),
    (
        "draw_sphere_mirror",
        include_str!("fixtures/sculpt_brushes/draw_sphere_mirror.json"),
    ),
    (
        "draw_sharp_sphere",
        include_str!("fixtures/sculpt_brushes/draw_sharp_sphere.json"),
    ),
    (
        "blob_sphere",
        include_str!("fixtures/sculpt_brushes/blob_sphere.json"),
    ),
    (
        "nudge_grid",
        include_str!("fixtures/sculpt_brushes/nudge_grid.json"),
    ),
    (
        "crease_sphere",
        include_str!("fixtures/sculpt_brushes/crease_sphere.json"),
    ),
    (
        "pinch_sphere",
        include_str!("fixtures/sculpt_brushes/pinch_sphere.json"),
    ),
    (
        "smooth_sphere",
        include_str!("fixtures/sculpt_brushes/smooth_sphere.json"),
    ),
    (
        "smooth_sphere_four_dabs",
        include_str!("fixtures/sculpt_brushes/smooth_sphere_four_dabs.json"),
    ),
    (
        "smooth_sphere_mirror",
        include_str!("fixtures/sculpt_brushes/smooth_sphere_mirror.json"),
    ),
    (
        "smooth_sphere_mirror_two",
        include_str!("fixtures/sculpt_brushes/smooth_sphere_mirror_two.json"),
    ),
    (
        "flatten_sphere",
        include_str!("fixtures/sculpt_brushes/flatten_sphere.json"),
    ),
    (
        "fill_sphere",
        include_str!("fixtures/sculpt_brushes/fill_sphere.json"),
    ),
    (
        "scrape_sphere",
        include_str!("fixtures/sculpt_brushes/scrape_sphere.json"),
    ),
    (
        "grab_grid",
        include_str!("fixtures/sculpt_brushes/grab_grid.json"),
    ),
    (
        "snake_hook_grid",
        include_str!("fixtures/sculpt_brushes/snake_hook_grid.json"),
    ),
    (
        "thumb_grid",
        include_str!("fixtures/sculpt_brushes/thumb_grid.json"),
    ),
    (
        "rotate_sphere",
        include_str!("fixtures/sculpt_brushes/rotate_sphere.json"),
    ),
    (
        "elastic_grab_grid",
        include_str!("fixtures/sculpt_brushes/elastic_grab_grid.json"),
    ),
    (
        "pose_grid",
        include_str!("fixtures/sculpt_brushes/pose_grid.json"),
    ),
    (
        "pose_grid_scale_translate",
        include_str!("fixtures/sculpt_brushes/pose_grid_scale_translate.json"),
    ),
    (
        "pose_grid_squash_stretch",
        include_str!("fixtures/sculpt_brushes/pose_grid_squash_stretch.json"),
    ),
    (
        "pose_grid_mirror",
        include_str!("fixtures/sculpt_brushes/pose_grid_mirror.json"),
    ),
    (
        "pose_grid_scale_translate_mirror",
        include_str!("fixtures/sculpt_brushes/pose_grid_scale_translate_mirror.json"),
    ),
    (
        "boundary_grid_bend",
        include_str!("fixtures/sculpt_brushes/boundary_grid_bend.json"),
    ),
    (
        "boundary_grid_grab",
        include_str!("fixtures/sculpt_brushes/boundary_grid_grab.json"),
    ),
    (
        "boundary_grid_expand",
        include_str!("fixtures/sculpt_brushes/boundary_grid_expand.json"),
    ),
    (
        "boundary_grid_inflate",
        include_str!("fixtures/sculpt_brushes/boundary_grid_inflate.json"),
    ),
    (
        "boundary_grid_twist",
        include_str!("fixtures/sculpt_brushes/boundary_grid_twist.json"),
    ),
    (
        "boundary_grid_smooth",
        include_str!("fixtures/sculpt_brushes/boundary_grid_smooth.json"),
    ),
    (
        "boundary_grid_bend_mirror",
        include_str!("fixtures/sculpt_brushes/boundary_grid_bend_mirror.json"),
    ),
    (
        "boundary_grid_grab_mirror",
        include_str!("fixtures/sculpt_brushes/boundary_grid_grab_mirror.json"),
    ),
];

fn number(value: &Value) -> Result<f64, Box<dyn Error>> {
    value
        .as_f64()
        .ok_or_else(|| format!("expected a number, found {value}").into())
}

fn vec3(value: &Value) -> Result<DVec3, Box<dyn Error>> {
    match value.as_array().map(Vec::as_slice) {
        Some([x, y, z]) => Ok(DVec3::new(number(x)?, number(y)?, number(z)?)),
        _ => Err(format!("expected a 3-vector, found {value}").into()),
    }
}

fn array<T>(
    value: &Value,
    mut parse: impl FnMut(&Value) -> Result<T, Box<dyn Error>>,
) -> Result<Vec<T>, Box<dyn Error>> {
    value
        .as_array()
        .ok_or_else(|| format!("expected an array, found {value}"))?
        .iter()
        .map(&mut parse)
        .collect()
}

/// Replays one fixture's stroke on its base mesh and returns the largest per-component difference
/// from Blender's result, after checking that the stroke moved something.
fn max_error_against_blender(name: &str, text: &str) -> Result<f64, Box<dyn Error>> {
    let document: Value = serde_json::from_str(text)?;
    let brush = match document["brush"].as_str() {
        Some("clay_strips") => DabBrush::ClayStrips,
        Some("inflate") => DabBrush::Inflate,
        Some("layer") => DabBrush::Layer,
        Some("draw") => DabBrush::Draw,
        Some("draw_sharp") => DabBrush::DrawSharp,
        Some("blob") => DabBrush::Blob,
        Some("nudge") => DabBrush::Nudge,
        Some("crease") => DabBrush::Crease,
        Some("pinch") => DabBrush::Pinch,
        Some("smooth") => DabBrush::Smooth,
        Some("flatten") => DabBrush::Flatten,
        Some("fill") => DabBrush::Fill,
        Some("scrape") => DabBrush::Scrape,
        Some("grab") => DabBrush::Grab,
        Some("snake_hook") => DabBrush::SnakeHook,
        Some("thumb") => DabBrush::Thumb,
        Some("rotate") => DabBrush::Rotate,
        Some("elastic_deform") => DabBrush::Elastic,
        Some("pose") => DabBrush::Pose(match document["deform"].as_str() {
            Some("ROTATE_TWIST") => PoseDeform::RotateTwist,
            Some("SCALE_TRANSLATE") => PoseDeform::ScaleTranslate,
            Some("SQUASH_STRETCH") => PoseDeform::SquashStretch,
            other => return Err(format!("{name}: unknown pose deform {other:?}").into()),
        }),
        Some("boundary") => DabBrush::Boundary(match document["deform"].as_str() {
            Some("BEND") => BoundaryDeform::Bend,
            Some("EXPAND") => BoundaryDeform::Expand,
            Some("INFLATE") => BoundaryDeform::Inflate,
            Some("GRAB") => BoundaryDeform::Grab,
            Some("TWIST") => BoundaryDeform::Twist,
            Some("SMOOTH") => BoundaryDeform::Smooth,
            other => return Err(format!("{name}: unknown boundary deform {other:?}").into()),
        }),
        other => return Err(format!("{name}: unknown brush {other:?}").into()),
    };
    // Blender's essentials falloff presets: Clay Strips SMOOTHER, Draw Sharp POW4, the rest SMOOTH.
    let falloff = match brush {
        DabBrush::ClayStrips => Falloff::Smoother,
        DabBrush::DrawSharp => Falloff::Pow4,
        _ => Falloff::Smooth,
    };
    let symmetry = if document["mirror"].as_bool() == Some(true) {
        vec!["x".to_owned()]
    } else {
        Vec::new()
    };
    let base = array(&document["base"], vec3)?;
    let faces = array(&document["faces"], |face| {
        array(face, |index| {
            let index = number(index)?;
            if index < 0.0 || index.fract() != 0.0 {
                return Err(format!("{name}: invalid vertex index {index}").into());
            }
            Ok(index as usize)
        })
    })?;
    let result = array(&document["result"], vec3)?;
    let dabs = array(&document["dabs"], |dab| {
        Ok((
            vec3(&dab["position"])?,
            number(&dab["pressure"])?,
            number(&dab["radius"])?,
            number(&dab["strength"])?,
        ))
    })?;
    let samples = dabs
        .into_iter()
        .enumerate()
        .map(
            |(index, (position, pressure, radius, strength))| StrokeSample {
                position,
                pressure,
                radius,
                strength,
                time: index as f64,
            },
        )
        .collect();
    let view = if document["view"].is_array() {
        vec3(&document["view"])?
    } else {
        DVec3::Z
    };
    let stroke = Stroke {
        brush,
        samples,
        falloff,
        symmetry,
        delta: None,
        // Each fixture records the view it was stroked from. Fixtures without one come from
        // Blender's TOP orthographic view (`generate_oracle.py` `set_top_ortho_view`), whose viewer
        // sits at +Z.
        view: Some(view),
    };
    let mut mesh = Mesh::from_positions_and_faces(base.clone(), faces)?;
    if document["mask"].is_array() {
        let values = array(&document["mask"], number)?;
        let mask: serde_json::Map<String, Value> = mesh
            .vertices
            .iter()
            .zip(values)
            .map(|(vertex, value)| (format!("v{}", vertex.id), json!(value)))
            .collect();
        mesh.attributes.insert(
            "sculpt.mask".to_owned(),
            json!({"domain": "vertices", "values": mask}),
        );
    }
    sculpt::apply_stroke(&mut mesh, &stroke)?;
    if mesh.vertices.len() != result.len() {
        return Err(format!("{name}: vertex count changed").into());
    }
    let moved = mesh
        .vertices
        .iter()
        .zip(&base)
        .any(|(vertex, before)| vertex.co != *before);
    if !moved {
        return Err(format!("{name}: the stroke moved no vertex").into());
    }
    Ok(mesh
        .vertices
        .iter()
        .zip(&result)
        .map(|(vertex, expected)| (vertex.co - *expected).abs().max_element())
        .fold(0.0, f64::max))
}

#[test]
fn dab_brushes_match_blender_5_2_2() -> Result<(), Box<dyn Error>> {
    for (name, text) in FIXTURES {
        let error = max_error_against_blender(name, text)?;
        assert!(
            error <= TOLERANCE,
            "{name}: max vertex error {error:e} exceeds {TOLERANCE:e}"
        );
    }
    Ok(())
}

fn create_scene() -> Result<(tempfile::TempDir, std::path::PathBuf), Box<dyn Error>> {
    let directory = tempfile::tempdir()?;
    let scene = directory.path().join("scene");
    let initialized = Command::new(env!("CARGO_BIN_EXE_pot"))
        .arg("init")
        .arg(&scene)
        .arg("--json")
        .output()?;
    assert!(initialized.status.success());
    Ok((directory, scene))
}

fn apply(scene: &Path, operations: &Value) -> Result<Value, Box<dyn Error>> {
    let file = scene.with_extension("operations.json");
    fs::write(
        &file,
        serde_json::to_vec(
            &json!({"schema_version": 1, "base_revision": 0, "operations": operations}),
        )?,
    )?;
    let output = Command::new(env!("CARGO_BIN_EXE_pot"))
        .arg("apply")
        .arg(scene)
        .arg("--file")
        .arg(file)
        .arg("--json")
        .output()?;
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stdout)
    );
    Ok(serde_json::from_slice(&output.stdout)?)
}

fn stored_body_mesh(scene: &Path) -> Result<Value, Box<dyn Error>> {
    let document: Value = serde_json::from_slice(&fs::read(scene.join("scene.json"))?)?;
    Ok(document["data_blocks"]["body_mesh"]["mesh"].clone())
}

/// The Clay Strips fixture's dabs through `sculpt.stroke` with no `falloff` field. The op's default
/// must be the Smoother curve Blender's Clay Strips brush uses, so it equals an explicit `smoother`
/// and differs from `smooth` on this stroke.
#[test]
fn clay_strips_op_without_falloff_defaults_to_smoother() -> Result<(), Box<dyn Error>> {
    let fixture: Value = serde_json::from_str(include_str!(
        "fixtures/sculpt_brushes/clay_strips_grid.json"
    ))?;
    let samples: Vec<Value> = fixture["dabs"]
        .as_array()
        .ok_or("fixture dabs must be an array")?
        .iter()
        .enumerate()
        .map(|(index, dab)| {
            json!({
                "position": dab["position"],
                "pressure": dab["pressure"],
                "radius": dab["radius"],
                "strength": dab["strength"],
                "time": index as f64
            })
        })
        .collect();
    let grid = json!({
        "op": "node.create", "id": "body", "kind": "grid",
        "params": {"size": 2.0, "x_subdivisions": 20, "y_subdivisions": 20}
    });
    let (_directory, scene) = create_scene()?;
    apply(&scene, &json!([grid.clone()]))?;
    let base = stored_body_mesh(&scene)?["vertices"].clone();

    let stroked = |falloff: Option<&str>| -> Result<Value, Box<dyn Error>> {
        let (_directory, scene) = create_scene()?;
        let mut stroke = json!({
            "op": "sculpt.stroke", "target": {"id": "body"}, "brush": "clay_strips",
            "scope": "shared", "samples": samples.clone()
        });
        if let Some(falloff) = falloff {
            stroke["falloff"] = json!(falloff);
        }
        apply(&scene, &json!([grid.clone(), stroke]))?;
        Ok(stored_body_mesh(&scene)?["vertices"].clone())
    };
    let default = stroked(None)?;
    let smoother = stroked(Some("smoother"))?;
    let smooth = stroked(Some("smooth"))?;
    assert_ne!(default, base, "the stroke must move vertices");
    assert_eq!(
        default, smoother,
        "omitted falloff must run the smoother curve"
    );
    assert_ne!(
        default, smooth,
        "smoother and smooth must differ on this stroke"
    );
    Ok(())
}
