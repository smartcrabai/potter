#![expect(clippy::unwrap_used, reason = "small sequencer rendering fixtures")]

use std::{error::Error, fs, path::Path};

use serde_json::json;
use tempfile::tempdir;

use super::{frame_output_names, render_camera_frame, render_vse_frame};
use crate::{
    eval::EvaluationContext,
    model::{Scene, SceneDoc},
    ops::apply_batch,
    sequencer::{EffectType, Strip, StripType, TransitionType},
};

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
    render_vse_frame(
        doc, scene, frame, width, height, root, "realtime", 1, 0, false,
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
    assert_pixel(text.pixels[1], [0.8, 0.4, 0.2, 0.75]);
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
        ..EvaluationContext::default()
    };
    let (_, raster) = render_camera_frame(
        &doc,
        &context,
        directory.path(),
        "camera_main",
        1,
        1,
        "beauty",
        "realtime",
        true,
        false,
        1,
        0,
    )?;
    let nested = render(&doc, &scene, directory.path(), 1.0, 1, 1)?;
    assert_eq!(nested.pixels, raster.linear_rgba);
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
