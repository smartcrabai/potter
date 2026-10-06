#![expect(clippy::unwrap_used, reason = "integration tests")]
#![expect(
    clippy::float_cmp,
    reason = "asserting exact deterministic animation values"
)]

use std::{collections::BTreeSet, error::Error, fs, io::Cursor, path::Path, process::Command};

use potter::{
    eval::{EvaluationContext, Snapshot},
    model::SceneDoc,
    ops::apply_batch,
};
use serde_json::{Value, json};
use tempfile::tempdir;

fn pot() -> Command {
    Command::new(env!("CARGO_BIN_EXE_pot"))
}

fn apply(doc: &SceneDoc, operations: &Value) -> SceneDoc {
    apply_batch(
        doc,
        &json!({"schema_version":1,"base_revision":doc.revision,"operations":operations}),
    )
    .unwrap()
    .doc
}

#[test]
fn nla_layers_blend_numerically_and_push_down_preserves_samples() {
    let mut doc = SceneDoc::new("00000000-0000-4000-8000-000000000000".to_owned());
    doc = apply(
        &doc,
        &json!([
            {"op":"node.create","id":"animated","kind":"empty"},
            {"op":"action.create","id":"ramp","name":"Ramp"},
            {"op":"action.update","target":{"id":"ramp"},"set":{"fcurves":[{"path":"transform.translation","index":0,"keyframes":[{"frame":1.0,"value":0.0},{"frame":11.0,"value":10.0}]}]}},
            {"op":"action.create","id":"offset","name":"Offset"},
            {"op":"action.update","target":{"id":"offset"},"set":{"fcurves":[{"path":"transform.translation","index":0,"keyframes":[{"frame":1.0,"value":4.0},{"frame":11.0,"value":4.0}]}]}},
            {"op":"nla.track_create","target":{"id":"animated"},"id":"track_main","name":"Main"},
            {"op":"nla.strip_create","target":{"id":"animated"},"track":"track_main","id":"strip_ramp","action":"ramp","frame_start":1.0,"frame_end":11.0,"action_frame_start":1.0,"action_frame_end":11.0,"scale":1.0,"repeat":1.0,"blend_type":"replace","influence":0.5,"extrapolation":"hold","blend_in":0.0,"blend_out":0.0},
            {"op":"nla.strip_create","target":{"id":"animated"},"track":"track_main","id":"strip_offset","action":"offset","frame_start":1.0,"frame_end":11.0,"action_frame_start":1.0,"action_frame_end":11.0,"scale":1.0,"repeat":1.0,"blend_type":"add","influence":0.5,"extrapolation":"hold","blend_in":0.0,"blend_out":0.0}
        ]),
    );
    let blended = Snapshot::evaluate(
        &doc,
        &EvaluationContext {
            frame: Some(6.0),
            ..EvaluationContext::default()
        },
    )
    .unwrap();
    assert!(
        (blended.nodes[&potter::model::Id::new("animated").unwrap()].world_matrix[12] - 4.5).abs()
            < 1.0e-9
    );
    doc = apply(
        &doc,
        &json!([
            {"op":"nla.track_update","target":{"id":"animated"},"track":"track_main","set":{"name":"Main Updated","solo":true}},
            {"op":"nla.strip_update","target":{"id":"animated"},"track":"track_main","strip":"strip_offset","set":{"influence":0.25}}
        ]),
    );
    let node_id = potter::model::Id::new("animated").unwrap();
    assert_eq!(doc.nodes[&node_id].nla_tracks[0].name, "Main Updated");
    let updated = Snapshot::evaluate(
        &doc,
        &EvaluationContext {
            frame: Some(6.0),
            ..EvaluationContext::default()
        },
    )
    .unwrap();
    assert!((updated.nodes[&node_id].world_matrix[12] - 3.5).abs() < 1.0e-9);
    doc = apply(
        &doc,
        &json!([{"op":"nla.strip_delete","target":{"id":"animated"},"track":"track_main","strip":"strip_offset"}]),
    );
    let after_strip_delete = Snapshot::evaluate(
        &doc,
        &EvaluationContext {
            frame: Some(6.0),
            ..EvaluationContext::default()
        },
    )
    .unwrap();
    assert!((after_strip_delete.nodes[&node_id].world_matrix[12] - 2.5).abs() < 1.0e-9);
    doc = apply(
        &doc,
        &json!([
            {"op":"nla.strip_delete","target":{"id":"animated"},"track":"track_main","strip":"strip_ramp"},
            {"op":"nla.track_delete","target":{"id":"animated"},"track":"track_main"}
        ]),
    );
    assert!(doc.nodes[&node_id].nla_tracks.is_empty());

    doc = apply(
        &doc,
        &json!([
            {"op":"action.create","id":"active","name":"Active"},
            {"op":"action.update","target":{"id":"active"},"set":{"fcurves":[{"path":"transform.translation","index":0,"keyframes":[{"frame":1.0,"value":2.0},{"frame":11.0,"value":12.0}]}]}},
            {"op":"node.update","target":{"id":"animated"},"set":{"action":"active"}},
            {"op":"action.slot_create","target":{"id":"active"},"id":"slot_primary","node":"animated"},
            {"op":"scene.marker_add","id":"marker_review","name":"Review","frame":6.0}
        ]),
    );
    let action_id = potter::model::Id::new("active").unwrap();
    assert_eq!(doc.actions[&action_id].slots[0].node.as_str(), "animated");
    assert_eq!(doc.scenes[&doc.active_scene].markers[0].frame, 6.0);
    let before = [1.0, 6.0, 11.0].map(|frame| {
        Snapshot::evaluate(
            &doc,
            &EvaluationContext {
                frame: Some(frame),
                ..EvaluationContext::default()
            },
        )
        .unwrap()
        .nodes[&potter::model::Id::new("animated").unwrap()]
            .world_matrix[12]
    });
    doc = apply(
        &doc,
        &json!([
            {"op":"nla.push_down","target":{"id":"animated"}},
            {"op":"scene.marker_remove","target":{"id":"marker_review"}}
        ]),
    );
    assert!(
        doc.nodes[&potter::model::Id::new("animated").unwrap()]
            .action
            .is_none()
    );
    assert!(doc.scenes[&doc.active_scene].markers.is_empty());
    let after = [1.0, 6.0, 11.0].map(|frame| {
        Snapshot::evaluate(
            &doc,
            &EvaluationContext {
                frame: Some(frame),
                ..EvaluationContext::default()
            },
        )
        .unwrap()
        .nodes[&potter::model::Id::new("animated").unwrap()]
            .world_matrix[12]
    });
    for (before, after) in before.into_iter().zip(after) {
        assert!((before - after).abs() < 1.0e-9);
    }
}

fn run_ok(mut command: Command) -> Result<std::process::Output, Box<dyn Error>> {
    let output = command.output()?;
    assert!(
        output.status.success(),
        "stderr: {}\nstdout: {}",
        String::from_utf8_lossy(&output.stderr),
        String::from_utf8_lossy(&output.stdout)
    );
    Ok(output)
}

fn init(scene: &Path) -> Result<(), Box<dyn Error>> {
    let mut command = pot();
    command.arg("init").arg(scene).arg("--json");
    run_ok(command)?;
    Ok(())
}

fn cli_apply(scene: &Path, temp: &Path, ops: &Value) -> Result<(), Box<dyn Error>> {
    let batch = temp.join("gp-ops.json");
    fs::write(
        &batch,
        serde_json::to_vec(&json!({"schema_version":1,"base_revision":0,"operations":ops}))?,
    )?;
    let mut command = pot();
    command
        .arg("apply")
        .arg(scene)
        .arg("--file")
        .arg(batch)
        .arg("--json");
    run_ok(command)?;
    Ok(())
}

#[test]
fn grease_pencil_svg_round_trip_and_pdf_xref_are_valid() -> Result<(), Box<dyn Error>> {
    let directory = tempdir()?;
    let source_scene = directory.path().join("source.pot");
    let imported_scene = directory.path().join("imported.pot");
    let svg = directory.path().join("drawing.svg");
    let pdf = directory.path().join("drawing.pdf");
    let preview = directory.path().join("previews");
    init(&source_scene)?;
    cli_apply(
        &source_scene,
        directory.path(),
        &json!([
            {"op":"node.create","id":"drawing","kind":"grease_pencil","name":"Drawing"},
            {"op":"grease_pencil.layer_create","target":{"id":"drawing"},"id":"ink","name":"Ink"},
            {"op":"grease_pencil.frame_add","target":{"id":"drawing"},"layer":"ink","frame":1.0},
            {"op":"grease_pencil.stroke_add","target":{"id":"drawing"},"layer":"ink","frame":1.0,"id":"stroke_main","points":[{"position":[-1.0,0.0,0.0],"pressure":1.0,"radius":0.05,"opacity":1.0,"time":0.0},{"position":[0.0,0.0,1.0],"pressure":0.8,"radius":0.05,"opacity":1.0,"time":0.5},{"position":[1.0,0.0,0.0],"pressure":0.7,"radius":0.05,"opacity":1.0,"time":1.0}],"material":null,"cyclic":false,"fill":null},
            {"op":"grease_pencil.layer_create","target":{"id":"drawing"},"id":"temporary_layer","name":"Temporary"},
            {"op":"grease_pencil.layer_update","target":{"id":"drawing"},"layer":"temporary_layer","set":{"name":"Temp"} },
            {"op":"grease_pencil.layer_delete","target":{"id":"drawing"},"layer":"temporary_layer"},
            {"op":"grease_pencil.stroke_add","target":{"id":"drawing"},"layer":"ink","frame":1.0,"id":"scratch","points":[{"position":[0.0,0.0,0.0],"pressure":1.0,"radius":0.01,"opacity":1.0,"time":0.0},{"position":[0.1,0.0,0.0],"pressure":1.0,"radius":0.01,"opacity":1.0,"time":1.0}]},
            {"op":"grease_pencil.stroke_delete","target":{"id":"drawing"},"layer":"ink","frame":1.0,"stroke":"scratch"},
            {"op":"grease_pencil.stroke_update","target":{"id":"drawing"},"layer":"ink","frame":1.0,"stroke":"stroke_main","set":{"cyclic":true}},
        ]),
    )?;
    let mut render_preview = pot();
    render_preview
        .arg("preview")
        .arg(&source_scene)
        .args(["--views", "front", "--size", "128", "--frame", "2", "--out"])
        .arg(&preview)
        .args(["--overwrite", "--json"]);
    run_ok(render_preview)?;
    let preview_bytes = fs::read(preview.join("front.png"))?;
    let mut png_reader = png::Decoder::new(Cursor::new(preview_bytes.as_slice())).read_info()?;
    let mut pixel_bytes = vec![
        0;
        png_reader
            .output_buffer_size()
            .ok_or("PNG output buffer size overflow")?
    ];
    let image_info = png_reader.next_frame(&mut pixel_bytes)?;
    let channels = match image_info.color_type {
        png::ColorType::Rgba => 4,
        png::ColorType::Rgb => 3,
        png::ColorType::GrayscaleAlpha => 2,
        png::ColorType::Grayscale => 1,
        png::ColorType::Indexed => {
            return Err(std::io::Error::other("preview PNG was not decoded").into());
        }
    };
    let distinct_pixels = pixel_bytes[..image_info.buffer_size()]
        .chunks_exact(channels)
        .map(<[u8]>::to_vec)
        .collect::<BTreeSet<_>>();
    assert!(
        distinct_pixels.len() > 1,
        "Grease Pencil preview contains no visible stroke pixels"
    );

    let mut export_svg = pot();
    export_svg
        .arg("export")
        .arg(&source_scene)
        .args(["--format", "svg", "--out"])
        .arg(&svg)
        .args(["--view", "front", "--frame", "2", "--json"]);
    run_ok(export_svg)?;
    let svg_text = fs::read_to_string(&svg)?;
    assert!(svg_text.contains("<svg"));
    assert!(
        svg_text.contains("stroke_main")
            || svg_text.contains("<polyline")
            || svg_text.contains("<path")
    );

    init(&imported_scene)?;
    let mut import_svg = pot();
    import_svg
        .arg("import")
        .arg(&imported_scene)
        .args(["--file"])
        .arg(&svg)
        .args([
            "--format",
            "svg",
            "--base-revision",
            "0",
            "--mode",
            "replace",
            "--json",
        ]);
    run_ok(import_svg)?;
    let imported = potter::store::Project::open(&imported_scene)?;
    let imported_gp = imported
        .doc()
        .data_blocks
        .values()
        .find_map(|data| data.grease_pencil.as_ref())
        .ok_or_else(|| std::io::Error::other("imported Grease Pencil data is missing"))?;
    let imported_stroke = imported_gp
        .layers
        .iter()
        .flat_map(|layer| &layer.frames)
        .flat_map(|frame| &frame.strokes)
        .next()
        .ok_or_else(|| std::io::Error::other("imported Grease Pencil stroke is missing"))?;
    assert_eq!(imported_stroke.points.len(), 3);
    let source = potter::store::Project::open(&source_scene)?;
    let source_gp = source
        .doc()
        .data_blocks
        .values()
        .find_map(|data| data.grease_pencil.as_ref())
        .ok_or_else(|| std::io::Error::other("source Grease Pencil data is missing"))?;
    let source_stroke = source_gp
        .layers
        .iter()
        .flat_map(|layer| &layer.frames)
        .flat_map(|frame| &frame.strokes)
        .next()
        .ok_or_else(|| std::io::Error::other("source Grease Pencil stroke is missing"))?;
    for (source_point, imported_point) in source_stroke.points.iter().zip(&imported_stroke.points) {
        assert!((source_point.position[0] - imported_point.position[0]).abs() < 1.0e-6);
        assert!((source_point.position[2] - imported_point.position[1]).abs() < 1.0e-6);
    }
    drop(source);

    let mut export_pdf = pot();
    export_pdf
        .arg("export")
        .arg(&source_scene)
        .args(["--format", "pdf", "--out"])
        .arg(&pdf)
        .args(["--view", "front", "--json"]);
    run_ok(export_pdf)?;
    let bytes = fs::read(&pdf)?;
    assert!(bytes.starts_with(b"%PDF-"));
    let startxref = b"startxref\n";
    let marker = bytes
        .windows(startxref.len())
        .rposition(|window| window == startxref)
        .ok_or_else(|| std::io::Error::other("PDF startxref is missing"))?;
    let offset_start = marker + startxref.len();
    let offset_end = offset_start
        + bytes[offset_start..]
            .iter()
            .position(|byte| *byte == b'\n')
            .ok_or_else(|| std::io::Error::other("PDF startxref value is unterminated"))?;
    let xref_offset = std::str::from_utf8(&bytes[offset_start..offset_end])?.parse::<usize>()?;
    let xref = bytes
        .get(xref_offset..)
        .ok_or_else(|| std::io::Error::other("PDF xref offset is outside the file"))?;
    assert!(xref.starts_with(b"xref\n0 5\n"));
    let mut entries = xref[b"xref\n0 5\n".len()..].split(|byte| *byte == b'\n');
    let _free = entries
        .next()
        .ok_or_else(|| std::io::Error::other("PDF xref free entry is missing"))?;
    for object_id in 1..=4 {
        let entry = entries
            .next()
            .ok_or_else(|| std::io::Error::other("PDF xref object entry is missing"))?;
        let object_offset = std::str::from_utf8(
            entry
                .get(..10)
                .ok_or_else(|| std::io::Error::other("PDF xref object entry is malformed"))?,
        )?
        .parse::<usize>()?;
        let object_header = format!("{object_id} 0 obj\n");
        assert!(
            bytes
                .get(object_offset..)
                .is_some_and(|value| value.starts_with(object_header.as_bytes()))
        );
    }
    Ok(())
}

#[test]
fn projection_option_is_restricted_to_svg_and_pdf() -> Result<(), Box<dyn Error>> {
    let directory = tempdir()?;
    let scene = directory.path().join("scene.pot");
    let output = directory.path().join("scene.obj");
    init(&scene)?;
    let mut command = pot();
    command
        .arg("export")
        .arg(&scene)
        .args(["--format", "obj", "--out"])
        .arg(&output)
        .args(["--view", "top", "--json"]);
    let result = command.output()?;
    assert!(!result.status.success());
    let response: Value = serde_json::from_slice(&result.stdout)?;
    assert_eq!(response["error"]["code"], "INVALID_ARGUMENT");
    Ok(())
}
