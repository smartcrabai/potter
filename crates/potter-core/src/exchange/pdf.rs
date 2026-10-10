use crate::{
    error::{ErrorCode, PotError, Result},
    eval::Snapshot,
    model::SceneDoc,
};
use std::fmt::Write as _;

use super::{
    render_visible_nodes,
    vector::{self, Bounds, Projection},
};

const PAGE_WIDTH: f64 = 612.0;
const PAGE_HEIGHT: f64 = 792.0;
const PAGE_MARGIN: f64 = 36.0;

struct StrokeDraw {
    points: Vec<[f64; 2]>,
    radii: Vec<f64>,
    color: [f64; 4],
    cyclic: bool,
}

#[derive(Clone, Copy)]
struct Frame {
    center: [f64; 2],
    scale: f64,
}

/// Export the evaluated scene as a single-page, vector PDF.
///
/// `view` accepts `front`, `top`, `right`, or `isometric`. Geometry is fit
/// proportionally inside a letter-sized page with fixed margins.
///
/// # Errors
///
/// Returns `INVALID_ARGUMENT` for an unknown view and `EVALUATION_FAILED` when
/// evaluated geometry contains coordinates or stroke data that cannot be represented.
pub fn export(doc: &SceneDoc, snapshot: &Snapshot, view: &str) -> Result<Vec<u8>> {
    let projection = parse_projection(view)?;
    let visible_nodes = render_visible_nodes(doc, snapshot)?;
    let mut bounds = Bounds::new();
    let mut mesh_lines = Vec::<[[f64; 2]; 2]>::new();
    vector::for_each_projected_mesh_edge(doc, snapshot, projection, |line| {
        for point in line {
            bounds.include(point, 0.0, "PDF geometry bounds are not finite")?;
        }
        mesh_lines.push(line);
        Ok(())
    })?;

    let mut strokes = Vec::new();
    vector::for_each_projected_stroke(doc, snapshot, &visible_nodes, projection, |projected| {
        let vector::ProjectedStroke {
            stroke,
            points,
            max_radius,
        } = projected;
        for point in &points {
            bounds.include(*point, max_radius, "PDF geometry bounds are not finite")?;
        }
        let radii = stroke.radius.iter().map(|radius| radius.max(0.0)).collect();
        strokes.push(StrokeDraw {
            points,
            radii,
            color: stroke.color,
            cyclic: stroke.cyclic,
        });
        Ok(())
    })?;

    let frame = frame_for(&bounds)?;
    let mut content = String::from("q\n1 J\n1 j\n");
    if !mesh_lines.is_empty() {
        content.push_str("0.16 0.16 0.16 RG\n0.55 w\n");
        for [from, to] in mesh_lines {
            let from = to_page(from, frame);
            let to = to_page(to, frame);
            let _ = writeln!(
                content,
                "{:.6} {:.6} m {:.6} {:.6} l S",
                from[0], from[1], to[0], to[1]
            );
        }
    }
    for stroke in &strokes {
        let color = pdf_color(stroke.color);
        let _ = writeln!(
            content,
            "{:.6} {:.6} {:.6} RG {:.6} {:.6} {:.6} rg",
            color[0], color[1], color[2], color[0], color[1], color[2]
        );
        draw_stroke(&mut content, stroke, frame);
    }
    content.push_str("Q\n");

    Ok(build_pdf(&content))
}

fn parse_projection(view: &str) -> Result<Projection> {
    match view {
        "front" => Ok(Projection::Front),
        "back" => Ok(Projection::Back),
        "top" => Ok(Projection::Top),
        "bottom" => Ok(Projection::Bottom),
        "right" => Ok(Projection::Right),
        "left" => Ok(Projection::Left),
        "iso" | "isometric" => Ok(Projection::Isometric),
        _ => Err(PotError::invalid_argument(format!(
            "unsupported PDF view `{view}`; expected front, back, right, left, top, bottom, or iso"
        ))),
    }
}

fn frame_for(bounds: &Bounds) -> Result<Frame> {
    let has_geometry = bounds.min[0].is_finite();
    if !has_geometry {
        return Ok(Frame {
            center: [0.0, 0.0],
            scale: 1.0,
        });
    }
    let width = bounds.max[0] - bounds.min[0];
    let height = bounds.max[1] - bounds.min[1];
    if !width.is_finite() || !height.is_finite() {
        return Err(PotError::new(
            ErrorCode::EvaluationFailed,
            "PDF geometry extent is not finite",
        ));
    }
    let safe_width = if width > f64::EPSILON { width } else { 1.0 };
    let safe_height = if height > f64::EPSILON { height } else { 1.0 };
    let scale = ((PAGE_WIDTH - 2.0 * PAGE_MARGIN) / safe_width)
        .min((PAGE_HEIGHT - 2.0 * PAGE_MARGIN) / safe_height);
    let center = [bounds.min[0] + width * 0.5, bounds.min[1] + height * 0.5];
    if !scale.is_finite() || scale <= 0.0 || center.iter().any(|coordinate| !coordinate.is_finite())
    {
        return Err(PotError::new(
            ErrorCode::EvaluationFailed,
            "PDF page framing is not finite",
        ));
    }
    Ok(Frame { center, scale })
}

fn to_page(point: [f64; 2], frame: Frame) -> [f64; 2] {
    [
        PAGE_WIDTH * 0.5 + (point[0] - frame.center[0]) * frame.scale,
        PAGE_HEIGHT * 0.5 + (point[1] - frame.center[1]) * frame.scale,
    ]
}

fn pdf_color(color: [f64; 4]) -> [f64; 3] {
    let alpha = color[3].clamp(0.0, 1.0);
    [
        (color[0].clamp(0.0, 1.0) * alpha) + 1.0 - alpha,
        (color[1].clamp(0.0, 1.0) * alpha) + 1.0 - alpha,
        (color[2].clamp(0.0, 1.0) * alpha) + 1.0 - alpha,
    ]
}

fn draw_stroke(content: &mut String, stroke: &StrokeDraw, frame: Frame) {
    let points = &stroke.points;
    if points.len() == 1 {
        let radius = stroke_radius(stroke, 0, frame.scale);
        let point = to_page(points[0], frame);
        draw_dot(content, point, radius);
        return;
    }
    let segments = if stroke.cyclic {
        points.len()
    } else {
        points.len() - 1
    };
    for index in 0..segments {
        let next = (index + 1) % points.len();
        let from = to_page(points[index], frame);
        let to = to_page(points[next], frame);
        let radius = f64::midpoint(
            stroke_radius(stroke, index, frame.scale),
            stroke_radius(stroke, next, frame.scale),
        );
        let _ = writeln!(content, "{:.6} w", (2.0 * radius).max(0.5));
        let _ = writeln!(
            content,
            "{:.6} {:.6} m {:.6} {:.6} l S",
            from[0], from[1], to[0], to[1]
        );
    }
}

fn stroke_radius(stroke: &StrokeDraw, index: usize, scale: f64) -> f64 {
    let radius = stroke
        .radii
        .get(index)
        .or_else(|| stroke.radii.last())
        .copied()
        .unwrap_or(0.0);
    (radius * scale).max(0.25)
}

fn draw_dot(content: &mut String, center: [f64; 2], radius: f64) {
    let kappa = 0.552_284_749_830_793_6;
    let right = center[0] + radius;
    let left = center[0] - radius;
    let top = center[1] + radius;
    let bottom = center[1] - radius;
    let _ = writeln!(content, "{right:.6} {:.6} m", center[1]);
    let _ = writeln!(
        content,
        "{right:.6} {:.6} {:.6} {top:.6} {:.6} {top:.6} c",
        center[1] + kappa * radius,
        center[0] + kappa * radius,
        center[0]
    );
    let _ = writeln!(
        content,
        "{:.6} {top:.6} {left:.6} {:.6} {left:.6} {:.6} c",
        center[0] - kappa * radius,
        center[1] + kappa * radius,
        center[1]
    );
    let _ = writeln!(
        content,
        "{left:.6} {:.6} {:.6} {bottom:.6} {:.6} {bottom:.6} c",
        center[1] - kappa * radius,
        center[0] - kappa * radius,
        center[0]
    );
    let _ = writeln!(
        content,
        "{:.6} {bottom:.6} {right:.6} {:.6} {right:.6} {:.6} c f",
        center[0] + kappa * radius,
        center[1] - kappa * radius,
        center[1]
    );
}

fn build_pdf(content: &str) -> Vec<u8> {
    let objects = [
        String::from("<< /Type /Catalog /Pages 2 0 R >>"),
        String::from("<< /Type /Pages /Kids [3 0 R] /Count 1 >>"),
        String::from(
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << >> /Contents 4 0 R >>",
        ),
        format!(
            "<< /Length {} >>\nstream\n{}\nendstream",
            content.len(),
            content
        ),
    ];
    let mut output = b"%PDF-1.4\n%\xE2\xE3\xCF\xD3\n".to_vec();
    let mut offsets = Vec::with_capacity(objects.len());
    for (index, object) in objects.iter().enumerate() {
        offsets.push(output.len());
        output.extend_from_slice(format!("{} 0 obj\n", index + 1).as_bytes());
        output.extend_from_slice(object.as_bytes());
        output.extend_from_slice(b"\nendobj\n");
    }
    let xref_offset = output.len();
    let mut xref = String::from("xref\n0 5\n0000000000 65535 f \n");
    for offset in offsets {
        let _ = writeln!(xref, "{offset:010} 00000 n ");
    }
    let _ = writeln!(xref, "trailer\n<< /Size 5 /Root 1 0 R >>");
    let _ = writeln!(xref, "startxref\n{xref_offset}\n%%EOF");
    output.extend_from_slice(xref.as_bytes());
    output
}
