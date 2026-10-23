#![expect(clippy::unwrap_used, reason = "small sequencer rendering fixtures")]

use std::{error::Error, fs, path::Path};

use serde_json::{Value, json};
use tempfile::tempdir;

use super::{frame_output_names, render_camera_frame, render_vse_frame};
use crate::{
    eval::{EvaluationContext, EvaluationPurpose, EvaluationSession},
    model::{Scene, SceneDoc},
    ops::apply_batch,
    sequencer::{EffectType, Strip, StripType, TransitionType},
};

fn camera_volume_doc(grid: Value, render_samples: u32) -> Result<SceneDoc, Box<dyn Error>> {
    const EYE_OFFSET: f64 = 0.2;
    let base = SceneDoc::new("00000000-0000-4000-8000-000000000000".to_owned());
    let outcome = apply_batch(
        &base,
        &json!({
            "schema_version": 1,
            "base_revision": 0,
            "operations": [
                {"op":"camera.create","id":"camera_main","transform":{"translation":[0,0,4]},"projection":"orthographic","ortho_scale":2},
                {"op":"camera.create","id":"camera_stereo","transform":{"translation":[0,0,4]},"projection":"orthographic","ortho_scale":2,"stereo_mode":"anaglyph","interocular_distance":EYE_OFFSET * 2.0},
                {"op":"camera.create","id":"camera_stereo_sbs","transform":{"translation":[0,0,4]},"projection":"orthographic","ortho_scale":2,"stereo_mode":"side_by_side","interocular_distance":EYE_OFFSET * 2.0},
                {"op":"camera.create","id":"camera_left","transform":{"translation":[-EYE_OFFSET,0,4]},"projection":"orthographic","ortho_scale":2},
                {"op":"camera.create","id":"camera_right","transform":{"translation":[EYE_OFFSET,0,4]},"projection":"orthographic","ortho_scale":2},
                {"op":"light.create","id":"emitter","light_type":"point","energy":100,"radius":2,"transform":{"translation":[0,0,0]}},
                {"op":"node.update","target":{"id":"emitter"},"set":{"ray_visibility":{"camera":true}}},
                {"op":"light.update","target":{"id":"emitter"},"set":{"volume_factor":0}},
                {"op":"volume.create","id":"fog","grids":[grid]},
                {"op":"scene.update","target":{"id":"scene_main"},"set":{"camera":"camera_main"}},
                {"op":"render.update","target":{"id":"scene_main"},"set":{"samples":render_samples,"engine":"realtime"}}
            ]
        }),
    )?;
    Ok(outcome.doc)
}

fn render_realtime_image(
    doc: &SceneDoc,
    root: &Path,
    camera_id: &str,
    width: u32,
    height: u32,
    samples: u32,
) -> Result<Vec<[f32; 4]>, Box<dyn Error>> {
    let context = EvaluationContext {
        scene_id: Some(doc.active_scene.clone()),
        frame: Some(1.0),
        purpose: EvaluationPurpose::Render,
        ..EvaluationContext::default()
    };
    let cache_directory = root.join(".potter/cache");
    let mut session = EvaluationSession::default();
    let (_, raster, _) = render_camera_frame(
        doc,
        &context,
        &mut session,
        root,
        &cache_directory,
        camera_id,
        width,
        height,
        "beauty",
        "realtime",
        false,
        samples,
        0,
    )?;
    Ok(raster.linear_rgba)
}

fn render_realtime_pixel(
    doc: &SceneDoc,
    root: &Path,
    camera_id: &str,
    samples: u32,
) -> Result<[f32; 4], Box<dyn Error>> {
    Ok(render_realtime_image(doc, root, camera_id, 1, 1, samples)?[0])
}

fn strip(id: &str, kind: StripType, channel: u32) -> Strip {
    Strip {
        id: id.to_owned(),
        kind,
        channel,
        frame_start: 1.0,
        length: 4.0,
        ..Strip::default()
    }
}

fn scene_with_strips(strips: Vec<Strip>) -> (SceneDoc, Scene) {
    let mut doc = SceneDoc::default();
    let scene_id = doc.active_scene.clone();
    doc.scenes.get_mut(&scene_id).unwrap().sequencer.strips = strips;
    let scene = doc.scenes.get(&scene_id).unwrap().clone();
    (doc, scene)
}

fn render(
    doc: &SceneDoc,
    scene: &Scene,
    root: &Path,
    frame: f64,
    width: u32,
    height: u32,
) -> crate::error::Result<crate::sequencer::RgbaFrame> {
    let cache_directory = root.join(".potter/cache");
    let mut session = EvaluationSession::default();
    let mut warnings = Vec::new();
    render_vse_frame(
        doc,
        scene,
        frame,
        width,
        height,
        root,
        &cache_directory,
        &mut session,
        "realtime",
        1,
        0,
        false,
        &mut warnings,
    )
}

fn color_strip(id: &str, channel: u32, color: [f32; 4]) -> Strip {
    Strip {
        color,
        ..strip(id, StripType::Color, channel)
    }
}

fn assert_pixel(actual: [f32; 4], expected: [f32; 4]) {
    for (actual, expected) in actual.into_iter().zip(expected) {
        assert!((actual - expected).abs() < 1.0e-5, "{actual} != {expected}");
    }
}

fn write_red_png(path: &Path) -> Result<(), Box<dyn Error>> {
    let file = fs::File::create(path)?;
    let mut encoder = png::Encoder::new(file, 1, 1);
    encoder.set_color(png::ColorType::Rgba);
    encoder.set_depth(png::BitDepth::Eight);
    let mut writer = encoder.write_header()?;
    writer.write_image_data(&[255, 0, 0, 255])?;
    Ok(())
}

#[test]
fn frame_filenames_use_frame_values_and_disambiguate_repeats() {
    assert_eq!(
        frame_output_names(&[1.0, 12.5, 24.0], "png").unwrap(),
        ["frame_0001.png", "frame_0012.5.png", "frame_0024.png"]
    );
    assert_eq!(
        frame_output_names(&[1.0, 1.0, 1.0], "exr").unwrap(),
        ["frame_0001.exr", "frame_0001__2.exr", "frame_0001__3.exr"]
    );
}

#[test]
fn color_text_image_image_sequence_and_meta_strips_render_numeric_pixels()
-> Result<(), Box<dyn Error>> {
    let directory = tempdir()?;
    let (doc, scene) = scene_with_strips(vec![color_strip("fill", 1, [0.25, 0.5, 0.75, 1.0])]);
    let color = render(&doc, &scene, directory.path(), 1.0, 1, 1)?;
    assert_pixel(color.pixels[0], [0.25, 0.5, 0.75, 1.0]);

    let (doc, scene) = scene_with_strips(vec![Strip {
        text: Some("A".to_owned()),
        color: [0.8, 0.4, 0.2, 0.75],
        ..strip("title", StripType::Text, 1)
    }]);
    let text = render(&doc, &scene, directory.path(), 1.0, 5, 7)?;
    assert_pixel(text.pixels[1], [0.6, 0.3, 0.15, 0.75]);
    assert_pixel(text.pixels[0], [0.0, 0.0, 0.0, 0.0]);

    write_red_png(&directory.path().join("still.png"))?;
    write_red_png(&directory.path().join("sequence_01.png"))?;
    for (kind, source, offset) in [
        (StripType::Image, "still.png", 0.0),
        (StripType::ImageSequence, "sequence_{frame:02}.png", 1.0),
    ] {
        let image_strip = Strip {
            source: Some(source.to_owned()),
            frame_offset_start: offset,
            length: 3.0,
            ..strip("image", kind, 1)
        };
        let (doc, scene) = scene_with_strips(vec![image_strip]);
        let image = render(&doc, &scene, directory.path(), 1.0, 1, 1)?;
        assert_pixel(image.pixels[0], [1.0, 0.0, 0.0, 1.0]);
    }

    let (doc, scene) = scene_with_strips(vec![
        color_strip("child", 1, [1.0, 0.0, 0.0, 1.0]),
        Strip {
            inputs: vec!["child".to_owned()],
            ..strip("meta", StripType::Meta, 2)
        },
    ]);
    let meta = render(&doc, &scene, directory.path(), 1.0, 1, 1)?;
    assert_pixel(meta.pixels[0], [1.0, 0.0, 0.0, 1.0]);

    let (doc, scene) = scene_with_strips(vec![Strip {
        source: Some("unused.wav".to_owned()),
        ..strip("sound", StripType::Sound, 1)
    }]);
    let sound_only = render(&doc, &scene, directory.path(), 1.0, 1, 1)?;
    assert_pixel(sound_only.pixels[0], [0.0, 0.0, 0.0, 1.0]);
    Ok(())
}

#[test]
fn scene_strips_match_their_source_camera_render() -> Result<(), Box<dyn Error>> {
    let document = SceneDoc::new("00000000-0000-4000-8000-000000000000".to_owned());
    let outcome = apply_batch(
        &document,
        &json!({
            "schema_version": 1,
            "base_revision": 0,
            "operations": [
                {"op":"camera.create","id":"camera_main","transform":{"translation":[0,0,4]},"projection":"orthographic","ortho_scale":3},
                {"op":"scene.update","target":{"id":"scene_main"},"set":{"camera":"camera_main"}}
            ]
        }),
    )?;
    let mut doc = outcome.doc;
    let scene_id = doc.active_scene.clone();
    let mut scene = doc.scenes.get(&scene_id).unwrap().clone();
    scene.sequencer.strips = vec![Strip {
        source: Some(scene_id.as_str().to_owned()),
        ..strip("nested", StripType::Scene, 1)
    }];
    doc.scenes.insert(scene_id.clone(), scene.clone());

    let directory = tempdir()?;
    let context = EvaluationContext {
        scene_id: Some(scene_id),
        frame: Some(1.0),
        purpose: EvaluationPurpose::Render,
        ..EvaluationContext::default()
    };
    let mut session = EvaluationSession::default();
    let (_, raster, _) = render_camera_frame(
        &doc,
        &context,
        &mut session,
        directory.path(),
        directory.path().join(".potter/cache").as_path(),
        "camera_main",
        1,
        1,
        "beauty",
        "realtime",
        false,
        1,
        0,
    )?;
    let nested = render(&doc, &scene, directory.path(), 1.0, 1, 1)?;
    assert_eq!(nested.pixels, raster.linear_rgba);
    Ok(())
}

#[test]
fn nested_realtime_volume_uses_outer_sequence_sample_count() -> Result<(), Box<dyn Error>> {
    let grid = json!({
        "dims": [5, 5, 3],
        "voxel_size": 1.0,
        "origin": [-2.0, -2.0, 1.0],
        "values": vec![1.0; 75]
    });
    let mut doc = camera_volume_doc(grid, 1)?;
    let scene_id = doc.active_scene.clone();
    let mut scene = doc.scenes.get(&scene_id).unwrap().clone();
    scene.sequencer.strips = vec![Strip {
        source: Some(scene_id.as_str().to_owned()),
        ..strip("nested_volume", StripType::Scene, 1)
    }];
    doc.scenes.insert(scene_id.clone(), scene.clone());

    const OUTER_SEQUENCE_SAMPLES: u32 = 4;
    assert_eq!(doc.scenes.get(&scene_id).unwrap().render.samples, 1);
    let directory = tempdir()?;
    let cache_directory = directory.path().join(".potter/cache");
    let mut session = EvaluationSession::default();
    let mut warnings = Vec::new();
    let output = render_vse_frame(
        &doc,
        &scene,
        1.0,
        1,
        1,
        directory.path(),
        &cache_directory,
        &mut session,
        "realtime",
        OUTER_SEQUENCE_SAMPLES,
        0,
        false,
        &mut warnings,
    )?;
    let pixel = output.pixels[0];
    assert!(pixel.into_iter().all(f32::is_finite), "{pixel:?}");
    assert!(
        pixel[..3].iter().any(|channel| *channel > 0.0),
        "camera-visible emitter through the nested volume rendered black: {pixel:?}"
    );
    Ok(())
}

#[test]
fn anaglyph_realtime_volume_resolves_each_eye_column_independently() -> Result<(), Box<dyn Error>> {
    const SAMPLES: u32 = 4;
    let values = (0..3)
        .flat_map(|_| (0..7).flat_map(|_| (0..7).map(|x| x as f32 * 0.8)))
        .collect::<Vec<_>>();
    let grid = json!({
        "dims": [7, 7, 3],
        "voxel_size": 0.5,
        "origin": [-1.5, -1.5, 1.0],
        "values": values
    });
    let doc = camera_volume_doc(grid, SAMPLES)?;
    let directory = tempdir()?;
    let anaglyph = render_realtime_pixel(&doc, directory.path(), "camera_stereo", SAMPLES)?;
    let left = render_realtime_pixel(&doc, directory.path(), "camera_left", SAMPLES)?;
    let right = render_realtime_pixel(&doc, directory.path(), "camera_right", SAMPLES)?;

    assert!(anaglyph.into_iter().all(f32::is_finite), "{anaglyph:?}");
    assert!(left.into_iter().all(f32::is_finite), "{left:?}");
    assert!(right.into_iter().all(f32::is_finite), "{right:?}");
    assert!(
        left[0] > right[0],
        "density increases toward the right eye: {left:?} vs {right:?}"
    );
    assert!(
        right[0] > 0.0,
        "the right eye must still see the emitter: {right:?}"
    );
    for (channel, expected) in [(0, left[0]), (1, right[1]), (2, right[2])] {
        assert!(
            (anaglyph[channel] - expected).abs() < 1.0e-5,
            "anaglyph channel {channel} is {}, expected independent-eye value {expected}; left={left:?}, right={right:?}",
            anaglyph[channel]
        );
    }
    assert!(
        anaglyph[0] > anaglyph[1],
        "the right-eye column was reused from the left eye: {anaglyph:?}"
    );
    Ok(())
}

#[test]
fn side_by_side_realtime_volume_odd_seam_matches_each_eye() -> Result<(), Box<dyn Error>> {
    const PACKED_WIDTH: u32 = 65;
    const SPLIT: u32 = PACKED_WIDTH / 2;
    const HEIGHT: u32 = 1;
    const SAMPLES: u32 = 1;

    let values = (0..3)
        .flat_map(|_| (0..7).flat_map(|_| (0..7).map(|x| x as f32 * 0.8)))
        .collect::<Vec<_>>();
    let grid = json!({
        "dims": [7, 7, 3],
        "voxel_size": 0.5,
        "origin": [-1.5, -1.5, 1.0],
        "values": values
    });
    let mut doc = camera_volume_doc(grid, SAMPLES)?;
    let scene_id = doc.active_scene.clone();
    doc.scenes
        .get_mut(&scene_id)
        .unwrap()
        .render
        .eevee_volumetric_tile_size = 16;

    let mut clear_doc = doc.clone();
    clear_doc
        .data_blocks
        .values_mut()
        .find_map(|block| block.volume.as_mut())
        .and_then(|volume| volume.grids.first_mut())
        .and_then(|grid| grid.values.as_mut())
        .unwrap()
        .fill(0.0);

    let directory = tempdir()?;
    let packed = render_realtime_image(
        &doc,
        directory.path(),
        "camera_stereo_sbs",
        PACKED_WIDTH,
        HEIGHT,
        SAMPLES,
    )?;
    let clear = render_realtime_image(
        &clear_doc,
        directory.path(),
        "camera_stereo_sbs",
        PACKED_WIDTH,
        HEIGHT,
        SAMPLES,
    )?;
    let left = render_realtime_image(
        &doc,
        directory.path(),
        "camera_left",
        SPLIT,
        HEIGHT,
        SAMPLES,
    )?;
    let right = render_realtime_image(
        &doc,
        directory.path(),
        "camera_right",
        PACKED_WIDTH - SPLIT,
        HEIGHT,
        SAMPLES,
    )?;

    let left_seam = packed[(SPLIT - 1) as usize];
    let right_seam = packed[SPLIT as usize];
    let clear_left_seam = clear[(SPLIT - 1) as usize];
    let clear_right_seam = clear[SPLIT as usize];
    for (actual, expected, description) in [
        (left_seam, left[(SPLIT - 1) as usize], "left-eye seam"),
        (right_seam, right[0], "right-eye seam"),
    ] {
        assert!(
            actual.into_iter().all(f32::is_finite),
            "{description}: {actual:?}"
        );
        assert!(
            actual[..3].iter().any(|channel| *channel > 0.0),
            "{description} lost the camera-visible emitter: {actual:?}"
        );
        for channel in 0..3 {
            assert!(
                (actual[channel] - expected[channel]).abs() < 1.0e-5,
                "{description} channel {channel} is {}, expected independent-eye value {}; left={left_seam:?}, right={right_seam:?}",
                actual[channel],
                expected[channel]
            );
        }
    }
    assert!(
        clear_left_seam[0] > left_seam[0],
        "the spatially varying volume must attenuate the left seam: {left_seam:?} vs {clear_left_seam:?}"
    );
    assert!(
        clear_right_seam[0] > right_seam[0],
        "the volume must attenuate the right seam: {right_seam:?} vs {clear_right_seam:?}"
    );
    Ok(())
}

#[test]
fn transitions_and_effect_strips_apply_numerical_operations() -> Result<(), Box<dyn Error>> {
    let directory = tempdir()?;
    for (transition, expected) in [
        (TransitionType::Cross, [0.5, 0.0, 0.5, 1.0]),
        (
            TransitionType::GammaCross,
            [0.735_356_9, 0.0, 0.735_356_9, 1.0],
        ),
    ] {
        let mut transition_strip = strip("transition", StripType::Transition, 3);
        transition_strip.length = 2.0;
        transition_strip.transition = Some(transition);
        transition_strip.inputs = vec!["red".to_owned(), "blue".to_owned()];
        let (doc, scene) = scene_with_strips(vec![
            color_strip("red", 1, [1.0, 0.0, 0.0, 1.0]),
            color_strip("blue", 2, [0.0, 0.0, 1.0, 1.0]),
            transition_strip,
        ]);
        let output = render(&doc, &scene, directory.path(), 2.0, 1, 1)?;
        assert_pixel(output.pixels[0], expected);
    }

    let mut wipe_strip = strip("wipe", StripType::Transition, 3);
    wipe_strip.length = 2.0;
    wipe_strip.transition = Some(TransitionType::Wipe);
    wipe_strip.inputs = vec!["red".to_owned(), "blue".to_owned()];
    let (doc, scene) = scene_with_strips(vec![
        color_strip("red", 1, [1.0, 0.0, 0.0, 1.0]),
        color_strip("blue", 2, [0.0, 0.0, 1.0, 1.0]),
        wipe_strip,
    ]);
    let wipe = render(&doc, &scene, directory.path(), 2.0, 2, 1)?;
    assert_pixel(wipe.pixels[0], [0.0, 0.0, 1.0, 1.0]);
    assert_pixel(wipe.pixels[1], [1.0, 0.0, 0.0, 1.0]);

    let effects = [
        (EffectType::Add, [0.3, 0.6, 0.9, 1.0], true),
        (EffectType::Subtract, [0.1, 0.2, 0.3, 1.0], true),
        (EffectType::Multiply, [0.02, 0.08, 0.18, 1.0], true),
        (EffectType::AlphaOver, [0.1, 0.2, 0.3, 1.0], true),
        (EffectType::Transform, [0.2, 0.4, 0.6, 1.0], false),
        (EffectType::Speed, [0.2, 0.4, 0.6, 1.0], false),
        (EffectType::Glow, [0.2, 0.4, 0.6, 1.0], false),
        (EffectType::GaussianBlur, [0.2, 0.4, 0.6, 1.0], false),
    ];
    for (effect, expected, two_inputs) in effects {
        let mut effect_strip = strip("effect", StripType::Effect, 3);
        effect_strip.effect = Some(effect);
        effect_strip.inputs = if two_inputs {
            vec!["first".to_owned(), "second".to_owned()]
        } else {
            vec!["first".to_owned()]
        };
        let mut strips = vec![color_strip("first", 1, [0.2, 0.4, 0.6, 1.0])];
        if two_inputs {
            strips.push(color_strip("second", 2, [0.1, 0.2, 0.3, 1.0]));
        }
        strips.push(effect_strip);
        let (doc, scene) = scene_with_strips(strips);
        let output = render(&doc, &scene, directory.path(), 2.0, 1, 1)?;
        assert_pixel(output.pixels[0], expected);
    }
    Ok(())
}

#[test]
fn unsupported_movie_and_unresolved_scene_strips_report_errors() -> Result<(), Box<dyn Error>> {
    let directory = tempdir()?;
    let (doc, scene) = scene_with_strips(vec![Strip {
        source: Some("movie.mov".to_owned()),
        ..strip("movie", StripType::Movie, 1)
    }]);
    assert_eq!(
        render(&doc, &scene, directory.path(), 1.0, 1, 1)
            .unwrap_err()
            .code,
        crate::error::ErrorCode::DependencyMissing
    );

    let (doc, scene) = scene_with_strips(vec![Strip {
        source: Some("missing_scene".to_owned()),
        ..strip("scene", StripType::Scene, 1)
    }]);
    assert_eq!(
        render(&doc, &scene, directory.path(), 1.0, 1, 1)
            .unwrap_err()
            .code,
        crate::error::ErrorCode::TargetNotFound
    );
    Ok(())
}

/// Potter's VSE `RgbaFrame` pixels are associated RGBA; the EXR writer stores them directly as Blender does. Reloading
/// the file gives the stored premultiplied channels, and `straight_pixels` recovers straight colour for nonzero alpha.
/// The Blender side uses the `exr32` premultiplied oracle in `tests/fixtures/alpha_texture`.
#[test]
fn potter_exr_writer_preserves_associated_alpha_and_alpha_zero_radiance()
-> Result<(), Box<dyn Error>> {
    let directory = tempdir()?;
    let path = directory.path().join("alpha_half_and_zero.exr");
    let associated = [[0.4, 0.2, 0.1, 0.5], [0.75, 0.5, 0.25, 0.0]];
    super::write_exr(&path, &associated, 2, 1)?;
    let (width, height, stored) =
        crate::image::decode_pixels(&fs::read(&path)?, crate::model::ImageColorspace::Linear)?;
    assert_eq!((width, height), (2, 1));
    for (channel, expected) in associated[0].into_iter().enumerate() {
        assert!(
            (stored[0][channel] - f64::from(expected)).abs() < 1.0e-6,
            "stored channel {channel} is {}, expected {expected}",
            stored[0][channel]
        );
    }
    for (channel, expected) in associated[1].into_iter().enumerate() {
        assert!(
            (stored[1][channel] - f64::from(expected)).abs() < 1.0e-6,
            "alpha-zero stored channel {channel} is {}, expected {expected}",
            stored[1][channel]
        );
    }
    let image = crate::image::ImageData {
        width,
        height,
        pixels: stored,
        tiles: std::collections::BTreeMap::new(),
        interpolation: crate::image::ImageInterpolation::Linear,
        srgb: false,
        udim: false,
        colour: crate::image::ColourCache::default(),
        storage: crate::image::ImageStorage {
            data: false,
            straight_alpha: false,
            ignore_alpha: false,
            precision: crate::image::Precision::Float,
        },
    };
    let straight = image.straight_pixels();
    for (channel, expected) in [0.8, 0.4, 0.2, 0.5].into_iter().enumerate() {
        assert!(
            (straight[0][channel] - expected).abs() < 1.0e-6,
            "straight channel {channel} is {}, expected {expected}",
            straight[0][channel]
        );
    }
    Ok(())
}
