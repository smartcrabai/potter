#![expect(clippy::unwrap_used, reason = "feature acceptance tests")]

use std::{fs, io::Cursor};

use potter::{
    eval::EvaluationContext,
    mask::{ControlPoint, Mask, MaskSpline},
    media::{AudioBuffer, decode_wav, encode_wav},
    model::SceneDoc,
    ops::apply_batch,
    render::render_sequence,
    sequencer::{AudioTrack, Strip, StripType, TransitionType, mix_audio},
};
use serde_json::{Value, json};

#[test]
fn sequencer_strips_are_editable_through_operations() {
    let document = SceneDoc::new("test".to_owned());
    let batch = json!({
        "schema_version": 1,
        "base_revision": 0,
        "operations": [{
            "op": "sequencer.strip_create",
            "id": "opening",
            "type": "color",
            "channel": 1,
            "frame_start": 1.0,
            "length": 24.0,
            "color": [0.2, 0.3, 0.4, 1.0]
        }]
    });
    let outcome = apply_batch(&document, &batch).unwrap();
    assert!(outcome.changed);
    assert_eq!(
        outcome.doc.scenes[&outcome.doc.active_scene]
            .sequencer
            .strips
            .len(),
        1
    );
    assert_eq!(
        outcome.doc.scenes[&outcome.doc.active_scene]
            .sequencer
            .strips[0]
            .id,
        "opening"
    );
}

#[test]
fn sequencer_strip_type_is_a_stable_serialized_contract() {
    let strip = Strip {
        id: "shot".to_owned(),
        kind: StripType::ImageSequence,
        ..Strip::default()
    };
    let value = serde_json::to_value(strip).unwrap();
    assert_eq!(value["type"], "image_sequence");
    assert!(value.get("kind").is_none());
}

#[test]
fn effect_strip_operations_preserve_inputs_and_reject_missing_inputs() {
    let document = SceneDoc::new("test".to_owned());
    let batch = json!({
        "schema_version": 1,
        "base_revision": 0,
        "operations": [
            {"op": "sequencer.strip_create", "id": "left", "type": "color", "color": [0.1, 0.2, 0.3, 1.0]},
            {"op": "sequencer.strip_create", "id": "right", "type": "color", "color": [0.4, 0.5, 0.6, 1.0]},
            {"op": "sequencer.strip_create", "id": "blend", "type": "effect", "effect": "add", "inputs": ["left", "right"]}
        ]
    });
    let outcome = apply_batch(&document, &batch).unwrap();
    let strips = &outcome.doc.scenes[&outcome.doc.active_scene]
        .sequencer
        .strips;
    let blend = strips.iter().find(|strip| strip.id == "blend").unwrap();
    assert_eq!(blend.effect, Some(potter::sequencer::EffectType::Add));
    assert_eq!(blend.inputs, vec!["left".to_owned(), "right".to_owned()]);

    let invalid = json!({
        "schema_version": 1,
        "base_revision": 0,
        "operations": [{
            "op": "sequencer.strip_create", "id": "blend", "type": "effect",
            "effect": "add", "inputs": ["left"]
        }]
    });
    let error = apply_batch(&document, &invalid).unwrap_err();
    assert_eq!(error.code, potter::error::ErrorCode::InvalidOperation);
}

#[test]
fn sequencer_strip_update_rejects_input_dependency_cycles() {
    let document = SceneDoc::new("test".to_owned());
    let create = json!({
        "schema_version": 1,
        "base_revision": 0,
        "operations": [
            {"op": "sequencer.strip_create", "id": "plate", "type": "color"},
            {"op": "sequencer.strip_create", "id": "transform", "type": "effect", "effect": "transform", "inputs": ["plate"]}
        ]
    });
    let document = apply_batch(&document, &create).unwrap().doc;
    let update = json!({
        "schema_version": 1,
        "base_revision": document.revision,
        "operations": [{
            "op": "sequencer.strip_update",
            "id": "plate",
            "set": {"type": "effect", "effect": "speed", "inputs": ["transform"]}
        }]
    });
    let error = apply_batch(&document, &update).unwrap_err();
    assert_eq!(error.code, potter::error::ErrorCode::InvalidOperation);
}

#[test]
fn strip_operations_reject_nonmonotonic_modifier_curves() {
    let document = SceneDoc::new("test".to_owned());
    let batch = json!({
        "schema_version": 1,
        "base_revision": 0,
        "operations": [{
            "op": "sequencer.strip_create",
            "id": "invalid_curve",
            "type": "color",
            "modifiers": [{"curves": [[1.0, 0.0], [0.0, 1.0]]}]
        }]
    });
    let error = apply_batch(&document, &batch).unwrap_err();
    assert_eq!(error.code, potter::error::ErrorCode::InvalidOperation);
}

#[test]
fn strip_delete_rejects_strips_with_live_effect_dependents() {
    let document = SceneDoc::new("test".to_owned());
    let create = json!({
        "schema_version": 1,
        "base_revision": 0,
        "operations": [
            {"op": "sequencer.strip_create", "id": "source", "type": "color"},
            {"op": "sequencer.strip_create", "id": "filter", "type": "effect", "effect": "glow", "inputs": ["source"]}
        ]
    });
    let document = apply_batch(&document, &create).unwrap().doc;
    let delete = json!({
        "schema_version": 1,
        "base_revision": document.revision,
        "operations": [{"op": "sequencer.strip_delete", "id": "source"}]
    });
    let error = apply_batch(&document, &delete).unwrap_err();
    assert_eq!(error.code, potter::error::ErrorCode::InvalidOperation);
}

#[test]
fn mask_operation_stores_a_rasterizable_spline() {
    let document = SceneDoc::new("test".to_owned());
    let batch = json!({
        "schema_version": 1,
        "base_revision": 0,
        "operations": [{
            "op": "mask.create",
            "id": "subject",
            "splines": [{
                "points": [
                    {"position": [2.0, 2.0], "keyframes": []},
                    {"position": [8.0, 2.0], "keyframes": []},
                    {"position": [8.0, 8.0], "keyframes": []},
                    {"position": [2.0, 8.0], "keyframes": []}
                ],
                "feather": 0.0
            }]
        }]
    });
    let outcome = apply_batch(&document, &batch).unwrap();
    let mask = outcome
        .doc
        .masks
        .get(&potter::model::Id::new("subject".to_owned()).unwrap())
        .unwrap()
        .clone();
    let image = mask.rasterize(10, 10, 1.0).unwrap();
    let area: f32 = image.pixels.iter().sum();
    assert!((area - 36.0).abs() <= 2.0);
}

#[test]
fn strip_create_rejects_nonfinite_timing() {
    let document = SceneDoc::new("test".to_owned());
    let mut operation = json!({
        "op": "sequencer.strip_create",
        "id": "bad",
        "type": "color",
        "length": 2.0
    });
    operation["frame_start"] = Value::String("NaN".to_owned());
    let batch = json!({"schema_version":1,"base_revision":0,"operations":[operation]});
    assert!(apply_batch(&document, &batch).is_err());
}

#[test]
fn audio_mix_converts_timeline_and_source_frame_offsets_at_their_rates() {
    let mut source_samples = vec![0.0; 1_000];
    source_samples.extend(vec![0.5; 2_000]);
    let source = AudioBuffer {
        sample_rate: 48_000,
        channels: 1,
        samples: source_samples,
    };
    let track = AudioTrack {
        audio: &source,
        frame_start: 3.5,
        frame_end: Some(3.75),
        frame_offset_start: 0.5,
        frame_offset_end: 0.0,
        retiming_keys: &[],
        volume: 1.0,
        pan: 0.0,
        pitch: 1.0,
    };
    let mixed = mix_audio(&[track], 3.5, 3.75, 24.0, 24_000).unwrap();
    assert_eq!(mixed.samples.len(), 250);
    assert!(
        mixed
            .samples
            .iter()
            .all(|sample| (*sample - 0.5).abs() < 1.0e-6)
    );
}

#[test]
fn audio_mix_follows_source_frame_retiming_keys() {
    let mut source_samples = vec![0.0; 2_000];
    source_samples.extend(vec![0.5; 2_000]);
    let source = AudioBuffer {
        sample_rate: 48_000,
        channels: 1,
        samples: source_samples,
    };
    let keys = [
        potter::sequencer::RetimingKey {
            frame: 0.0,
            source_frame: 1.0,
        },
        potter::sequencer::RetimingKey {
            frame: 1.0,
            source_frame: 2.0,
        },
    ];
    let track = AudioTrack {
        audio: &source,
        frame_start: 0.0,
        frame_end: Some(1.0),
        frame_offset_start: 0.0,
        frame_offset_end: 0.0,
        retiming_keys: &keys,
        volume: 1.0,
        pan: 0.0,
        pitch: 1.0,
    };
    let mixed = mix_audio(&[track], 0.0, 1.0, 24.0, 48_000).unwrap();
    assert!(
        mixed
            .samples
            .iter()
            .all(|sample| (*sample - 0.5).abs() < 1.0e-6)
    );
}

#[test]
fn polygon_mask_reference_area_is_geometric() {
    let mask = Mask::new(vec![
        MaskSpline::from_control_points(
            vec![
                ControlPoint::new(glam::DVec2::new(2.0, 2.0)),
                ControlPoint::new(glam::DVec2::new(8.0, 2.0)),
                ControlPoint::new(glam::DVec2::new(8.0, 8.0)),
                ControlPoint::new(glam::DVec2::new(2.0, 8.0)),
            ],
            0.0,
        )
        .unwrap(),
    ])
    .unwrap();
    let image = mask.rasterize(10, 10, 1.0).unwrap();
    let area: f32 = image.pixels.iter().sum();
    assert!((area - 36.0).abs() <= 2.0);
}

#[test]
fn strip_frame_bounds_and_retiming_map_timeline_to_source_frames() {
    let strip = Strip {
        id: "retimed".to_owned(),
        frame_start: 10.0,
        length: 8.0,
        retiming_keys: vec![
            potter::sequencer::RetimingKey {
                frame: 10.0,
                source_frame: 5.0,
            },
            potter::sequencer::RetimingKey {
                frame: 14.0,
                source_frame: 13.0,
            },
        ],
        ..Strip::default()
    };
    assert!(strip.contains_frame(10.0));
    assert!(strip.contains_frame(17.999));
    assert!(!strip.contains_frame(18.0));
    assert_eq!(strip.source_frame(12.0), 9.0);
}

#[test]
fn sequencer_render_writes_crossfade_frame_and_frame_aligned_audio() {
    let directory = tempfile::tempdir().unwrap();
    let source_audio_path = directory.path().join("source.wav");
    let source_audio = AudioBuffer {
        sample_rate: 48_000,
        channels: 2,
        samples: vec![0.25; 4_000 * 2],
    };
    fs::write(&source_audio_path, encode_wav(&source_audio).unwrap()).unwrap();

    let mut document = SceneDoc::new("00000000-0000-4000-8000-000000000000".to_owned());
    let scene = document.scenes.get_mut(&document.active_scene).unwrap();
    scene.fps = 24;
    scene.fps_base = 1.0;
    scene.render.use_sequencer = true;
    scene.sequencer.strips = vec![
        Strip {
            id: "red".to_owned(),
            kind: StripType::Color,
            channel: 1,
            frame_start: 1.0,
            length: 2.0,
            color: [1.0, 0.0, 0.0, 1.0],
            ..Strip::default()
        },
        Strip {
            id: "blue".to_owned(),
            kind: StripType::Color,
            channel: 2,
            frame_start: 1.0,
            length: 2.0,
            color: [0.0, 0.0, 1.0, 1.0],
            ..Strip::default()
        },
        Strip {
            id: "crossfade".to_owned(),
            kind: StripType::Transition,
            channel: 3,
            frame_start: 1.0,
            length: 2.0,
            transition: Some(TransitionType::Cross),
            inputs: vec!["red".to_owned(), "blue".to_owned()],
            ..Strip::default()
        },
        Strip {
            id: "sound".to_owned(),
            kind: StripType::Sound,
            channel: 4,
            frame_start: 2.0,
            length: 1.0,
            source: Some(source_audio_path.display().to_string()),
            sound_volume: 0.5,
            ..Strip::default()
        },
    ];
    let result = render_sequence(
        &document,
        &EvaluationContext::default(),
        directory.path(),
        None,
        &[2.0],
        "png",
        &directory.path().join("render"),
        false,
        "realtime",
        1,
        1,
        1,
        0,
        false,
    )
    .unwrap();

    let frame_path = result["frames"][0]["path"].as_str().unwrap();
    let frame_bytes = fs::read(frame_path).unwrap();
    let decoder = png::Decoder::new(Cursor::new(frame_bytes));
    let mut reader = decoder.read_info().unwrap();
    let mut pixels = vec![0_u8; reader.output_buffer_size().unwrap()];
    let info = reader.next_frame(&mut pixels).unwrap();
    assert_eq!(info.width, 1);
    assert_eq!(info.height, 1);
    assert_eq!(info.color_type, png::ColorType::Rgba);
    assert!((i16::from(pixels[0]) - 188).abs() <= 1);
    assert_eq!(pixels[1], 0);
    assert!((i16::from(pixels[2]) - 188).abs() <= 1);
    assert_eq!(pixels[3], 255);

    let audio_path = result["audio"].as_str().unwrap();
    let mixed = decode_wav(&fs::read(audio_path).unwrap()).unwrap();
    assert_eq!(mixed.sample_rate, 48_000);
    assert_eq!(mixed.channels, 2);
    assert_eq!(mixed.samples.len(), 2_000 * 2);
    assert!((mixed.samples[0] - 0.125).abs() < 0.001);
}
