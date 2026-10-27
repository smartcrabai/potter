//! Blender 5.2.2 oracle for Dynamic Paint IMAGE outputs and Potter bake sequences.

use std::{
    collections::BTreeMap,
    error::Error,
    fs, io,
    path::{Path, PathBuf},
    process::{Command, Output},
};

use serde_json::Value;
use tempfile::tempdir;

#[path = "common/blender_file.rs"]
mod blender_file;
use blender_file::blender_executable;

const CAPTURE: &str = include_str!("fixtures/dynamic_paint_image/capture.json");
type TestResult<T> = Result<T, Box<dyn Error>>;

fn pot() -> Command {
    Command::new(env!("CARGO_BIN_EXE_pot"))
}

fn checked_output(command: &mut Command, label: &str) -> TestResult<Output> {
    let output = command.output()?;
    if !output.status.success() {
        return Err(io::Error::other(format!(
            "{label} failed: stdout={} stderr={}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr),
        ))
        .into());
    }
    Ok(output)
}

fn run_pot_json(command: &mut Command, label: &str) -> TestResult<Value> {
    let output = checked_output(command, label)?;
    Ok(serde_json::from_slice(&output.stdout)?)
}

fn rgba8_pixels(path: &Path, width: u32, height: u32) -> TestResult<Vec<u8>> {
    let image = image::open(path)
        .map_err(|error| io::Error::other(format!("{}: {error}", path.display())))?;
    assert_eq!(image.width(), width, "{} width", path.display());
    assert_eq!(image.height(), height, "{} height", path.display());
    Ok(image.into_rgba8().into_raw())
}

fn assert_painted_pixel(pixels: &[u8], width: u32, column: u32, row: u32, label: &str) {
    let start = ((row * width + column) * 4) as usize;
    let pixel = &pixels[start..start + 4];
    assert!(
        pixel[3] > 0 && (pixel[0] > 0 || pixel[1] > 0 || pixel[2] > 0),
        "{label} must paint a colored pixel at ({column}, {row})"
    );
}

fn assert_scenario_discriminator(
    id: &str,
    pixels: &[u8],
    width: u32,
    height: u32,
    frame: u64,
    label: &str,
) {
    match id {
        "uv_edge_v01_filename_patterns" => {
            assert_eq!((width, height), (20, 20));
            let column = if frame == 1 { 0 } else { width - 1 };
            for blender_row in [0, 1] {
                let png_row = height - 1 - blender_row;
                assert_painted_pixel(
                    pixels,
                    width,
                    column,
                    png_row,
                    &format!("{label} Blender row {blender_row}"),
                );
            }
        }
        "uv_collinear_default_outputs" => {
            assert_eq!((width, height), (20, 20));
            for row in [9, 10] {
                assert_painted_pixel(pixels, width, 10, row, label);
            }
        }
        _ => {}
    }
}

fn read_file(path: &Path) -> TestResult<Vec<u8>> {
    fs::read(path).map_err(|error| io::Error::other(format!("{}: {error}", path.display())).into())
}

fn painted_pixel_count(pixels: &[u8]) -> usize {
    pixels
        .as_chunks::<4>()
        .0
        .iter()
        .filter(|pixel| pixel[3] != 0 && (pixel[0] != 0 || pixel[1] != 0 || pixel[2] != 0))
        .count()
}
fn assert_uv_outer_margin(pixels: &[u8], label: &str, frame: u64) {
    for row in 0..16 {
        for column in [0, 15] {
            assert_eq!(
                pixels[(row * 16 + column) * 4 + 3],
                0,
                "{label} frame {frame} painted through the UV island margin at ({column}, {row})"
            );
        }
    }
    for column in 0..16 {
        for row in [0, 15] {
            assert_eq!(
                pixels[(row * 16 + column) * 4 + 3],
                0,
                "{label} frame {frame} painted through the UV island margin at ({column}, {row})"
            );
        }
    }
}
fn assert_uv_island_margin_pixel(pixels: &[u8], label: &str, frame: u64) {
    let (column, row) = if frame == 1 { (2, 7) } else { (13, 7) };
    let alpha = pixels[(row * 16 + column) * 4 + 3];
    assert!(
        alpha > 0,
        "{label} frame {frame} must paint its touched island-margin pixel ({column}, {row})"
    );
}

#[test]
fn dynamic_paint_image_bake_matches_blender_png_sequence() -> TestResult<()> {
    let Some(blender) = blender_executable() else {
        eprintln!("skipping Dynamic Paint IMAGE oracle; Blender is unavailable");
        return Ok(());
    };

    let fixture =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/dynamic_paint_image");
    let expected_capture: Value = serde_json::from_str(CAPTURE)?;
    assert_eq!(expected_capture["blender_version"], "5.2.2 LTS");
    let scenarios = expected_capture["scenarios"]
        .as_array()
        .ok_or("committed Blender scenarios are missing")?;
    let scenario_ids = scenarios
        .iter()
        .map(|scenario| scenario["id"].as_str())
        .collect::<Option<Vec<_>>>()
        .ok_or("scenario id is missing")?;
    assert_eq!(
        scenario_ids,
        vec![
            "point",
            "volume_antialias_on",
            "volume_antialias_off",
            "drying_dissolve",
            "uv_islands_margin",
            "uv_edge_v01_filename_patterns",
            "uv_collinear_default_outputs",
        ]
    );
    let mut volume_aa_on = None;
    let mut volume_aa_off = None;
    let mut point_wetness = None;
    let mut drying_dissolve_wetness = None;

    let directory = tempdir()?;
    let generated = directory.path().join("generated");
    let generator = fixture.join("generate.py");
    checked_output(
        Command::new(&blender)
            .args(["-b", "--factory-startup", "--threads", "1", "--python"])
            .arg(generator)
            .arg("--")
            .arg(&generated),
        "Blender Dynamic Paint IMAGE oracle generator",
    )?;

    let generated_capture: Value =
        serde_json::from_slice(&read_file(&generated.join("capture.json"))?)?;
    assert_eq!(
        generated_capture, expected_capture,
        "Blender's generated Dynamic Paint IMAGE capture differs from committed fixture"
    );

    for scenario in scenarios {
        let id = scenario["id"].as_str().ok_or("scenario id is missing")?;
        let fixture_case = fixture.join(
            scenario["directory"]
                .as_str()
                .ok_or("scenario fixture directory is missing")?,
        );
        let generated_case = generated.join(
            scenario["directory"]
                .as_str()
                .ok_or("scenario generated directory is missing")?,
        );
        let source_blend = fixture_case.join(
            scenario["blend"]
                .as_str()
                .ok_or("scenario Blender file is missing")?,
        );
        assert!(source_blend.is_file(), "{id} Blender file is missing");
        assert!(
            generated_case
                .join(
                    scenario["blend"]
                        .as_str()
                        .ok_or("scenario Blender file is missing")?
                )
                .is_file(),
            "{id} generated Blender file is missing"
        );

        let expected_surface = &scenario["surface"];
        let expected_source = scenario["brush_source"]
            .as_str()
            .ok_or("scenario brush source is missing")?;
        match id {
            "point" => assert_eq!(expected_source, "POINT"),
            "volume_antialias_on" => {
                assert_eq!(expected_source, "VOLUME");
                assert_eq!(expected_surface["use_antialiasing"], true);
            }
            "volume_antialias_off" => {
                assert_eq!(expected_source, "VOLUME");
                assert_eq!(expected_surface["use_antialiasing"], false);
            }
            "drying_dissolve" => {
                assert_eq!(expected_surface["use_drying"], true);
                assert_eq!(expected_surface["use_dissolve"], true);
            }
            "uv_islands_margin" => {
                assert_eq!(scenario["mesh"]["uv_island_count"], 2);
                assert_eq!(scenario["mesh"]["uv_margin_pixels"], 2);
                assert_eq!(scenario["mesh"]["brush_touches_island_edges"], true);
            }
            "uv_edge_v01_filename_patterns" => {
                assert_eq!(scenario["mesh"]["uv_min_v"], 0.1);
                assert_eq!(expected_surface["image_resolution"], 20);
            }
            "uv_collinear_default_outputs" => {
                assert_eq!(scenario["mesh"]["uv_sample_v"], 0.5);
                assert_eq!(scenario["mesh"]["uv_face_is_collinear"], true);
                assert_eq!(expected_surface["image_resolution"], 20);
            }
            _ => unreachable!("scenario ids are checked above"),
        }
        let (expected_output_name_a, expected_output_name_b) = match id {
            "volume_antialias_on" => ("dp_paintmap##", "dp_wetmap"),
            "uv_edge_v01_filename_patterns" => ("paint_####.png", "paint_##.jpg"),
            "uv_collinear_default_outputs" => ("paintmap", "wetmap"),
            _ => ("dp_paintmap", "dp_wetmap"),
        };
        assert_eq!(expected_surface["output_name_a"], expected_output_name_a);
        assert_eq!(expected_surface["output_name_b"], expected_output_name_b);
        let expected_resolution = u32::try_from(
            expected_surface["image_resolution"]
                .as_u64()
                .ok_or("image resolution is missing")?,
        )?;

        let outputs = scenario["outputs"]
            .as_array()
            .ok_or("committed Blender image outputs are missing")?;
        assert_eq!(outputs.len(), 4, "{id} expects A/B images for both frames");
        if id == "uv_edge_v01_filename_patterns" {
            let files = outputs
                .iter()
                .map(|output| output["file"].as_str())
                .collect::<Option<Vec<_>>>()
                .ok_or("normalized PNG filenames are missing")?;
            assert_eq!(
                files,
                vec![
                    "paint_0001.png",
                    "paint_0002.png",
                    "paint_01.png",
                    "paint_02.png"
                ]
            );
        }
        let image_dir = scenario["image_dir"]
            .as_str()
            .ok_or("scenario image directory is missing")?;
        let mut blender_pixels = BTreeMap::new();
        for output in outputs {
            let name = output["name"]
                .as_str()
                .ok_or("Blender image output name is missing")?;
            let frame = output["frame"]
                .as_u64()
                .ok_or("Blender image output frame is missing")?;
            let filename = output["file"]
                .as_str()
                .ok_or("Blender image output filename is missing")?;
            let width = u32::try_from(
                output["width"]
                    .as_u64()
                    .ok_or("Blender image width is missing")?,
            )?;
            let height = u32::try_from(
                output["height"]
                    .as_u64()
                    .ok_or("Blender image height is missing")?,
            )?;
            assert_eq!(width, expected_resolution, "{id} image width");
            assert_eq!(height, expected_resolution, "{id} image height");
            let generated_pixels = rgba8_pixels(
                &generated_case.join(image_dir).join(filename),
                width,
                height,
            )?;
            let committed_pixels =
                rgba8_pixels(&fixture_case.join(image_dir).join(filename), width, height)?;
            assert_eq!(
                generated_pixels, committed_pixels,
                "generated Blender pixels differ from committed {id}/{filename}"
            );
            if name == expected_surface["output_name_a"] {
                assert!(
                    painted_pixel_count(&generated_pixels) > 0,
                    "Blender's {id} {name}, frame {frame} contains no painted pixels"
                );
                if id == "uv_islands_margin" {
                    assert_uv_outer_margin(&generated_pixels, "Blender UV islands", frame);
                    assert_uv_island_margin_pixel(&generated_pixels, "Blender UV islands", frame);
                }
                assert_scenario_discriminator(
                    id,
                    &generated_pixels,
                    width,
                    height,
                    frame,
                    &format!("Blender {id} frame {frame}"),
                );
            }
            blender_pixels.insert((name.to_owned(), frame), generated_pixels);
        }

        for name in [
            expected_surface["output_name_a"]
                .as_str()
                .ok_or("A output name is missing")?,
            expected_surface["output_name_b"]
                .as_str()
                .ok_or("B output name is missing")?,
        ] {
            let frame_one = blender_pixels
                .get(&(name.to_owned(), 1))
                .ok_or("Blender frame 1 output is missing")?;
            let frame_two = blender_pixels
                .get(&(name.to_owned(), 2))
                .ok_or("Blender frame 2 output is missing")?;
            if id != "uv_collinear_default_outputs" {
                assert_ne!(
                    frame_one, frame_two,
                    "{id} {name} must change between frames"
                );
            }
        }
        for frame in [1, 2] {
            let paint = blender_pixels
                .get(&(
                    expected_surface["output_name_a"]
                        .as_str()
                        .ok_or("A output name is missing")?
                        .to_owned(),
                    frame,
                ))
                .ok_or("A output is missing")?;
            let wetness = blender_pixels
                .get(&(
                    expected_surface["output_name_b"]
                        .as_str()
                        .ok_or("B output name is missing")?
                        .to_owned(),
                    frame,
                ))
                .ok_or("B output is missing")?;
            assert_ne!(paint, wetness, "{id} A and B must differ at frame {frame}");
        }

        let project = directory.path().join(format!("project-{id}"));
        checked_output(
            pot().args(["init"]).arg(&project).arg("--json"),
            &format!("pot init Dynamic Paint IMAGE {id} project"),
        )?;
        run_pot_json(
            pot()
                .args(["import"])
                .arg(&project)
                .args(["--file"])
                .arg(&source_blend)
                .args([
                    "--format",
                    "blend",
                    "--allow-lossy",
                    "--mode",
                    "replace",
                    "--base-revision",
                    "0",
                    "--blender",
                ])
                .arg(&blender)
                .arg("--json"),
            &format!("pot import Dynamic Paint IMAGE {id} fixture"),
        )?;

        let scene: Value = serde_json::from_slice(&read_file(&project.join("scene.json"))?)?;
        let nodes = scene["nodes"]
            .as_object()
            .ok_or("imported project nodes are missing")?;
        let (canvas_id, canvas) = nodes
            .iter()
            .find(|(_, node)| node["name"] == scenario["canvas"])
            .ok_or("imported Dynamic Paint IMAGE canvas is missing")?;
        let surfaces = canvas["properties"]["physics_dynamic_paint"]["canvas_surfaces"]
            .as_array()
            .ok_or("imported Dynamic Paint canvas surfaces are missing")?;
        assert_eq!(
            surfaces.len(),
            1,
            "{id} fixture has one active canvas surface"
        );
        let surface = &surfaces[0];
        for key in [
            "surface_type",
            "surface_format",
            "is_active",
            "frame_start",
            "frame_end",
            "use_antialiasing",
            "use_drying",
            "dry_speed",
            "use_dissolve",
            "dissolve_speed",
            "output_name_a",
            "use_output_a",
            "output_name_b",
            "use_output_b",
        ] {
            assert_eq!(surface[key], expected_surface[key], "{id} surface {key}");
        }
        assert_eq!(surface["surface_format"], "IMAGE");
        assert!(surface["is_active"].as_bool().is_some_and(|active| active));
        assert!(
            surface["image_output_path"]
                .as_str()
                .is_none_or(str::is_empty),
            "IMAGE output path must be supplied by the bake destination"
        );
        let (_, brush) = nodes
            .iter()
            .find(|(_, node)| node["name"] == scenario["brush"])
            .ok_or("imported Dynamic Paint IMAGE brush is missing")?;
        assert_eq!(
            brush["properties"]["physics_dynamic_paint"]["paint_source"], scenario["brush_source"],
            "{id} brush source"
        );

        let canvas_id = canvas_id.as_str();
        let inspected = run_pot_json(
            pot()
                .args(["inspect"])
                .arg(&project)
                .args(["--id", canvas_id, "--json"]),
            &format!("pot inspect Dynamic Paint IMAGE {id} canvas"),
        )?;
        assert!(
            inspected["result"]["items"]
                .as_array()
                .is_some_and(|items| items.iter().any(|item| item["id"] == canvas_id)),
            "pot inspect omitted the imported IMAGE canvas"
        );

        let bake_dir = directory.path().join(format!("bake-{id}"));
        let bake = run_pot_json(
            pot()
                .args(["bake"])
                .arg(&project)
                .args(["--kind", "simulation", "--frames", "1:2", "--out"])
                .arg(&bake_dir)
                .arg("--json"),
            &format!("pot bake Dynamic Paint IMAGE {id} frames 1:2"),
        )?;
        assert_eq!(bake["result"]["frame_count"], 2);

        let mut potter_pixels = BTreeMap::new();
        for output in outputs {
            let name = output["name"].as_str().ok_or("output name is missing")?;
            let frame = output["frame"].as_u64().ok_or("output frame is missing")?;
            let filename = output["file"].as_str().ok_or("output file is missing")?;
            let path = bake_dir.join(image_dir).join(filename);
            assert!(path.is_file(), "Potter image output is missing: {path:?}");
            let actual = rgba8_pixels(&path, expected_resolution, expected_resolution)?;
            let expected = blender_pixels
                .get(&(name.to_owned(), frame))
                .ok_or("Blender image pixels are missing")?;
            if name == expected_surface["output_name_a"] {
                assert!(
                    painted_pixel_count(&actual) > 0,
                    "Potter's {id} {name}, frame {frame} contains no painted pixels"
                );
                if id == "uv_islands_margin" {
                    assert_uv_outer_margin(&actual, "Potter UV islands", frame);
                    assert_uv_island_margin_pixel(&actual, "Potter UV islands", frame);
                }
                assert_scenario_discriminator(
                    id,
                    &actual,
                    expected_resolution,
                    expected_resolution,
                    frame,
                    &format!("Potter {id} frame {frame}"),
                );
            }
            assert_eq!(
                &actual, expected,
                "Potter pixels differ from Blender for {id} {name}, frame {frame}"
            );
            potter_pixels.insert((name.to_owned(), frame), actual);
        }
        assert_eq!(potter_pixels.len(), 4);
        for name in [
            expected_surface["output_name_a"]
                .as_str()
                .ok_or("A output name is missing")?,
            expected_surface["output_name_b"]
                .as_str()
                .ok_or("B output name is missing")?,
        ] {
            let frame_one = potter_pixels
                .get(&(name.to_owned(), 1))
                .ok_or("Potter frame 1 output is missing")?;
            let frame_two = potter_pixels
                .get(&(name.to_owned(), 2))
                .ok_or("Potter frame 2 output is missing")?;
            if id != "uv_collinear_default_outputs" {
                assert_ne!(
                    frame_one, frame_two,
                    "Potter {id} {name} must change between frames"
                );
            }
        }
        for frame in [1, 2] {
            let paint = potter_pixels
                .get(&(
                    expected_surface["output_name_a"]
                        .as_str()
                        .ok_or("A output name is missing")?
                        .to_owned(),
                    frame,
                ))
                .ok_or("A output is missing")?;
            let wetness = potter_pixels
                .get(&(
                    expected_surface["output_name_b"]
                        .as_str()
                        .ok_or("B output name is missing")?
                        .to_owned(),
                    frame,
                ))
                .ok_or("B output is missing")?;
            assert_ne!(
                paint, wetness,
                "Potter {id} A and B must differ at frame {frame}"
            );
        }
        let output_a = expected_surface["output_name_a"]
            .as_str()
            .ok_or("A output name is missing")?;
        let output_b = expected_surface["output_name_b"]
            .as_str()
            .ok_or("B output name is missing")?;
        if id == "volume_antialias_on" {
            volume_aa_on = Some(
                potter_pixels
                    .remove(&(output_a.to_owned(), 1))
                    .ok_or("Potter volume AA-on output is missing")?,
            );
        } else if id == "volume_antialias_off" {
            volume_aa_off = Some(
                potter_pixels
                    .remove(&(output_a.to_owned(), 1))
                    .ok_or("Potter volume AA-off output is missing")?,
            );
        }
        if id == "point" {
            point_wetness = Some(
                potter_pixels
                    .remove(&(output_b.to_owned(), 2))
                    .ok_or("Potter POINT wetness output is missing")?,
            );
        } else if id == "drying_dissolve" {
            drying_dissolve_wetness = Some(
                potter_pixels
                    .remove(&(output_b.to_owned(), 2))
                    .ok_or("Potter drying/dissolve wetness output is missing")?,
            );
        }
    }
    let volume_aa_on = volume_aa_on.ok_or("Potter volume AA-on paint output is missing")?;
    let volume_aa_off = volume_aa_off.ok_or("Potter volume AA-off paint output is missing")?;
    assert_ne!(
        volume_aa_on, volume_aa_off,
        "Potter's VOLUME AA-on and AA-off outputs must differ"
    );
    let point_wetness = point_wetness.ok_or("Potter POINT wetness output is missing")?;
    let drying_dissolve_wetness =
        drying_dissolve_wetness.ok_or("Potter drying/dissolve wetness output is missing")?;
    assert_ne!(
        point_wetness, drying_dissolve_wetness,
        "Potter's drying/dissolve settings must change the wetness IMAGE output"
    );
    Ok(())
}
