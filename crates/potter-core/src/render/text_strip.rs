use std::path::Path;

use ab_glyph_rasterizer::{Point, Rasterizer, point};
use serde_json::json;
use ttf_parser::{Face, GlyphId, OutlineBuilder, Tag};

use crate::{
    error::{ErrorCode, PotError, Result},
    eval::resolve_text_font_bytes,
    model::SceneDoc,
    sequencer::RgbaFrame,
};

const FONT_SIZE_PX: u32 = 60;
const MAX_TEXT_CHARACTERS: usize = 16_384;
const MAX_FONT_BYTES: usize = 64 * 1024 * 1024;
const MAX_GLYPH_DIMENSION: usize = 4_096;
const MAX_COVERAGE_PIXELS: usize = 64_000_000;

#[derive(Clone, Copy)]
struct TextGlyph {
    glyph: Option<GlyphId>,
    advance: f32,
    blf_width: f32,
    position_x: f32,
    whitespace: bool,
    break_space: bool,
    line_end: bool,
}

#[derive(Default)]
struct TextLine {
    glyphs: Vec<TextGlyph>,
    width: f32,
}

pub(super) fn render_external_text_frame(
    doc: &SceneDoc,
    project_root: &Path,
    font_reference: &str,
    text: &str,
    wrap_width: f32,
    color: [f32; 4],
    resolution_percentage: u32,
    width: u32,
    height: u32,
) -> Result<RgbaFrame> {
    if font_reference.is_empty() || font_reference == "builtin" {
        return Err(unsupported("an explicit static external font is required"));
    }
    if width == 0 || height == 0 || width > 16_384 || height > 16_384 {
        return Err(PotError::new(
            ErrorCode::InvalidArgument,
            "text strip render dimensions must be between 1 and 16384 pixels",
        ));
    }
    if text.chars().take(MAX_TEXT_CHARACTERS + 1).count() > MAX_TEXT_CHARACTERS {
        return Err(PotError::new(
            ErrorCode::LimitExceeded,
            "text strip content exceeds the supported character budget",
        ));
    }

    let Some((font_bytes, _)) = resolve_text_font_bytes(doc, Some(project_root), font_reference)?
    else {
        return Err(unsupported("the external font could not be resolved"));
    };
    if font_bytes.len() > MAX_FONT_BYTES {
        return Err(PotError::new(
            ErrorCode::LimitExceeded,
            "external text font exceeds the supported file size",
        ));
    }
    let face = Face::parse(&font_bytes, 0).map_err(|_| {
        unsupported("font is not a supported static TrueType/OpenType outline font")
    })?;
    validate_face(&face)?;
    let scaled_size = f64::from(FONT_SIZE_PX) * f64::from(resolution_percentage) / 100.0;
    if !scaled_size.is_finite() || scaled_size < 0.0 || scaled_size > f64::from(u32::MAX) {
        return Err(PotError::new(
            ErrorCode::LimitExceeded,
            "scaled text font size exceeds the supported range",
        ));
    }
    let font_size_px = scaled_size.trunc() as u32;
    let width_usize = usize::try_from(width).map_err(|_| {
        PotError::new(
            ErrorCode::LimitExceeded,
            "text frame width exceeds platform limits",
        )
    })?;
    let height_usize = usize::try_from(height).map_err(|_| {
        PotError::new(
            ErrorCode::LimitExceeded,
            "text frame height exceeds platform limits",
        )
    })?;
    let pixel_count = width_usize
        .checked_mul(height_usize)
        .ok_or_else(|| PotError::new(ErrorCode::LimitExceeded, "text frame dimensions overflow"))?;
    let mut pixels = Vec::new();
    pixels
        .try_reserve_exact(pixel_count)
        .map_err(|_| PotError::new(ErrorCode::LimitExceeded, "text frame allocation failed"))?;
    pixels.resize(pixel_count, [0.0; 4]);
    if font_size_px == 0 {
        return RgbaFrame::new(width, height, pixels);
    }

    let units_per_em = f32::from(face.units_per_em());
    if units_per_em <= 0.0 {
        return Err(unsupported("font has invalid em metrics"));
    }
    let scale = font_size_px as f32 / units_per_em;
    let frame_width = width as f32;
    let available_width = if wrap_width == 0.0 {
        i32::MAX as f32
    } else {
        (wrap_width * frame_width) as i32 as f32
    };
    let lines = layout_text(&face, text, scale, available_width)?;
    let line_height = ((f32::from(face.height()) + f32::from(face.line_gap())) * scale)
        .round()
        .max(1.0);
    let descender = (f32::from(face.descender()) * scale).floor();
    if !line_height.is_finite() || !descender.is_finite() {
        return Err(unsupported(
            "font metrics exceed the supported text layout range",
        ));
    }
    let line_advance = line_height;
    let block_height = line_height * lines.len() as f32;
    let image_center_y = height as f32 * 0.5;
    let first_baseline_from_bottom =
        (image_center_y - line_height - descender + block_height * 0.5).round();
    let first_baseline = height as f32 - first_baseline_from_bottom;

    let mut coverage_pixels = 0_usize;
    for (line_index, line) in lines.iter().enumerate() {
        let line_x = ((frame_width - line.width) * 0.5).round();
        let baseline = first_baseline + line_index as f32 * line_advance;
        for glyph in &line.glyphs {
            let Some(glyph_id) = glyph.glyph else {
                continue;
            };
            draw_glyph(
                &face,
                glyph_id,
                glyph.whitespace,
                line_x + glyph.position_x,
                baseline,
                scale,
                color,
                width_usize,
                height_usize,
                &mut pixels,
                &mut coverage_pixels,
            )?;
        }
    }
    RgbaFrame::new(width, height, pixels)
}
fn validate_face(face: &Face<'_>) -> Result<()> {
    if face.is_variable() || face.raw_face().table(Tag::from_bytes(b"fvar")).is_some() {
        return Err(unsupported("variable external fonts are not supported"));
    }
    if face.tables().glyf.is_none() && face.tables().cff.is_none() {
        return Err(unsupported(
            "font does not contain a supported TrueType or CFF outline",
        ));
    }
    if !face.tables().cmap.is_some_and(|cmap| {
        cmap.subtables
            .into_iter()
            .any(|subtable| subtable.is_unicode())
    }) {
        return Err(unsupported("font has no Unicode character map"));
    }
    let raw = face.raw_face();
    if [
        b"COLR", b"CPAL", b"CBDT", b"CBLC", b"EBDT", b"EBLC", b"bdat", b"bloc", b"sbix", b"SVG ",
        b"CFF2",
    ]
    .into_iter()
    .any(|tag| raw.table(Tag::from_bytes(tag)).is_some())
    {
        return Err(unsupported(
            "color glyphs and CFF2 outlines are not supported",
        ));
    }
    Ok(())
}

fn layout_text(
    face: &Face<'_>,
    text: &str,
    scale: f32,
    available_width: f32,
) -> Result<Vec<TextLine>> {
    let character_count = text.chars().count();
    let mut characters = Vec::new();
    characters
        .try_reserve_exact(character_count)
        .map_err(|_| PotError::new(ErrorCode::LimitExceeded, "text layout allocation failed"))?;

    for character in text.chars() {
        if character == '\n' {
            characters.push(TextGlyph {
                glyph: None,
                advance: 0.0,
                blf_width: 0.0,
                position_x: 0.0,
                whitespace: false,
                break_space: false,
                line_end: true,
            });
            continue;
        }
        if character == '\r' {
            characters.push(TextGlyph {
                glyph: None,
                advance: 0.0,
                blf_width: 0.0,
                position_x: 0.0,
                whitespace: true,
                break_space: false,
                line_end: false,
            });
            continue;
        }
        if is_unsupported_format_or_emoji_component(character) {
            return Err(unsupported(
                "Unicode format controls and emoji sequence components are not supported",
            ));
        }
        if is_combining_mark(character) {
            return Err(unsupported("combining marks are not supported"));
        }
        if !is_simple_ltr_character(character) {
            return Err(unsupported(
                "right-to-left or complex-script text is not supported",
            ));
        }
        let glyph = face
            .glyph_index(character)
            .filter(|glyph| glyph.0 != 0)
            .ok_or_else(|| {
                unsupported(&format!(
                    "font has no glyph for U+{:04X}",
                    u32::from(character)
                ))
            })?;
        let advance = face
            .glyph_hor_advance(glyph)
            .ok_or_else(|| unsupported("font has no horizontal glyph metrics"))?;
        let raw_advance = f32::from(advance) * scale;
        let advance = raw_advance.round();
        if !advance.is_finite() || (advance == 0.0 && !character.is_whitespace()) {
            return Err(unsupported("font has invalid horizontal glyph metrics"));
        }
        let bounds = face.glyph_bounding_box(glyph);
        if bounds.is_none() && !character.is_whitespace() {
            return Err(unsupported("font glyph has no supported outline"));
        }
        let blf_width = bounds.map_or(raw_advance.ceil(), |bounds| {
            let min_x = f32::from(bounds.x_min) * scale;
            let max_x = f32::from(bounds.x_max) * scale;
            raw_advance.max(max_x).ceil() - min_x.min(0.0).floor()
        });
        characters.push(TextGlyph {
            glyph: Some(glyph),
            advance,
            blf_width,
            position_x: 0.0,
            whitespace: character.is_whitespace(),
            break_space: character == ' ',
            line_end: false,
        });
    }

    let mut current_x = 0.0_f32;
    let mut last_space: Option<(usize, f32)> = None;
    for index in 0..characters.len() {
        if characters[index].line_end {
            current_x = 0.0;
            last_space = None;
            continue;
        }
        if characters[index].glyph.is_none() {
            continue;
        }
        if characters[index].break_space {
            last_space = Some((index, current_x));
        } else if current_x + characters[index].advance > available_width
            && let Some((space_index, space_x)) = last_space
        {
            characters[space_index].line_end = true;
            current_x -= space_x + characters[space_index].advance;
            last_space = None;
        }
        current_x += characters[index].advance;
    }

    let mut lines = Vec::new();
    lines
        .try_reserve(1)
        .map_err(|_| PotError::new(ErrorCode::LimitExceeded, "text layout allocation failed"))?;
    lines.push(TextLine::default());
    let mut current_x = 0.0_f32;
    for mut glyph in characters {
        glyph.position_x = current_x;
        current_x += glyph.advance;
        let line_end = glyph.line_end;
        let line = lines.last_mut().ok_or_else(|| {
            PotError::new(
                ErrorCode::InternalError,
                "text layout lost its current line",
            )
        })?;
        line.glyphs.try_reserve(1).map_err(|_| {
            PotError::new(ErrorCode::LimitExceeded, "text layout allocation failed")
        })?;
        line.glyphs.push(glyph);
        if line_end {
            set_line_width(line, true);
            lines.try_reserve(1).map_err(|_| {
                PotError::new(ErrorCode::LimitExceeded, "text layout allocation failed")
            })?;
            lines.push(TextLine::default());
            current_x = 0.0;
        }
    }
    if let Some(line) = lines.last_mut() {
        set_line_width(line, false);
    }
    Ok(lines)
}

fn set_line_width(line: &mut TextLine, exclude_last: bool) {
    let visible_count = line.glyphs.len().saturating_sub(usize::from(exclude_last));
    let Some(last) = visible_count
        .checked_sub(1)
        .and_then(|index| line.glyphs.get(index))
    else {
        line.width = 0.0;
        return;
    };
    line.width = last.position_x + last.blf_width.ceil();
}
fn draw_glyph(
    face: &Face<'_>,
    glyph: GlyphId,
    whitespace: bool,
    pen_x: f32,
    baseline: f32,
    scale: f32,
    color: [f32; 4],
    frame_width: usize,
    frame_height: usize,
    pixels: &mut [[f32; 4]],
    coverage_pixels: &mut usize,
) -> Result<()> {
    let Some(bounds) = face.glyph_bounding_box(glyph) else {
        return if whitespace {
            Ok(())
        } else {
            Err(unsupported("font glyph has no supported outline"))
        };
    };
    let left_f = pen_x + f32::from(bounds.x_min) * scale;
    let right_f = pen_x + f32::from(bounds.x_max) * scale;
    let top_f = baseline - f32::from(bounds.y_max) * scale;
    let bottom_f = baseline - f32::from(bounds.y_min) * scale;
    if ![left_f, right_f, top_f, bottom_f]
        .into_iter()
        .all(f32::is_finite)
    {
        return Err(PotError::new(
            ErrorCode::LimitExceeded,
            "font glyph bounds exceed the supported range",
        ));
    }
    let left = checked_pixel(left_f.floor())?;
    let right = checked_pixel(right_f.ceil())?;
    let top = checked_pixel(top_f.floor())?;
    let bottom = checked_pixel(bottom_f.ceil())?;
    if right <= 0 || bottom <= 0 || left >= frame_width as i64 || top >= frame_height as i64 {
        return Ok(());
    }
    let glyph_width = usize::try_from(right.saturating_sub(left)).map_err(|_| {
        PotError::new(
            ErrorCode::LimitExceeded,
            "font glyph width exceeds platform limits",
        )
    })?;
    let glyph_height = usize::try_from(bottom.saturating_sub(top)).map_err(|_| {
        PotError::new(
            ErrorCode::LimitExceeded,
            "font glyph height exceeds platform limits",
        )
    })?;
    if glyph_width == 0 || glyph_height == 0 {
        return Ok(());
    }
    if glyph_width > MAX_GLYPH_DIMENSION || glyph_height > MAX_GLYPH_DIMENSION {
        return Err(PotError::new(
            ErrorCode::LimitExceeded,
            "font glyph exceeds the rasterization budget",
        ));
    }
    let glyph_pixels = glyph_width.checked_mul(glyph_height).ok_or_else(|| {
        PotError::new(
            ErrorCode::LimitExceeded,
            "font glyph rasterization dimensions overflow",
        )
    })?;
    *coverage_pixels = coverage_pixels.checked_add(glyph_pixels).ok_or_else(|| {
        PotError::new(
            ErrorCode::LimitExceeded,
            "text rasterization dimensions overflow",
        )
    })?;
    if *coverage_pixels > MAX_COVERAGE_PIXELS {
        return Err(PotError::new(
            ErrorCode::LimitExceeded,
            "text exceeds the rasterization budget",
        ));
    }

    let mut rasterizer = Rasterizer::new(glyph_width, glyph_height);
    let mut outline = GlyphOutline {
        rasterizer: &mut rasterizer,
        scale,
        pen_x,
        baseline,
        left: left as f32,
        top: top as f32,
        current: None,
        contour_start: None,
    };
    if face.outline_glyph(glyph, &mut outline).is_none() {
        return Err(unsupported("font glyph has no supported outline"));
    }
    rasterizer.for_each_pixel_2d(|x, y, coverage| {
        let destination_x = left + x as i64;
        let destination_y = top + y as i64;
        if destination_x >= 0
            && destination_y >= 0
            && destination_x < frame_width as i64
            && destination_y < frame_height as i64
            && coverage > 0.0
        {
            let (Ok(destination_x), Ok(destination_y)) = (
                usize::try_from(destination_x),
                usize::try_from(destination_y),
            ) else {
                return;
            };
            let index = destination_y * frame_width + destination_x;
            let coverage_8 = (coverage.clamp(0.0, 1.0) * 255.0).round() as u8;
            let coverage = f32::from(coverage_8) / 255.0;
            let coverage_alpha = color[3] * coverage;
            let alpha = pixels[index][3] + coverage_alpha * (1.0 - pixels[index][3]);
            pixels[index] = [color[0], color[1], color[2], alpha];
        }
    });
    Ok(())
}

struct GlyphOutline<'a> {
    rasterizer: &'a mut Rasterizer,
    scale: f32,
    pen_x: f32,
    baseline: f32,
    left: f32,
    top: f32,
    current: Option<Point>,
    contour_start: Option<Point>,
}

impl GlyphOutline<'_> {
    fn transform(&self, x: f32, y: f32) -> Point {
        point(
            self.pen_x + x * self.scale - self.left,
            self.baseline - y * self.scale - self.top,
        )
    }

    fn close_contour(&mut self) {
        if let (Some(current), Some(start)) = (self.current.take(), self.contour_start.take()) {
            self.rasterizer.draw_line(current, start);
        }
    }
}

impl OutlineBuilder for GlyphOutline<'_> {
    fn move_to(&mut self, x: f32, y: f32) {
        self.close_contour();
        let point = self.transform(x, y);
        self.current = Some(point);
        self.contour_start = Some(point);
    }

    fn line_to(&mut self, x: f32, y: f32) {
        let end = self.transform(x, y);
        if let Some(start) = self.current {
            self.rasterizer.draw_line(start, end);
        }
        self.current = Some(end);
    }

    fn quad_to(&mut self, x1: f32, y1: f32, x: f32, y: f32) {
        let control = self.transform(x1, y1);
        let end = self.transform(x, y);
        if let Some(start) = self.current {
            self.rasterizer.draw_quad(start, control, end);
        }
        self.current = Some(end);
    }

    fn curve_to(&mut self, x1: f32, y1: f32, x2: f32, y2: f32, x: f32, y: f32) {
        let control1 = self.transform(x1, y1);
        let control2 = self.transform(x2, y2);
        let end = self.transform(x, y);
        if let Some(start) = self.current {
            self.rasterizer.draw_cubic(start, control1, control2, end);
        }
        self.current = Some(end);
    }

    fn close(&mut self) {
        self.close_contour();
    }
}

fn checked_pixel(value: f32) -> Result<i64> {
    if !value.is_finite() || value < i32::MIN as f32 || value > i32::MAX as f32 {
        return Err(PotError::new(
            ErrorCode::LimitExceeded,
            "font glyph bounds exceed the supported range",
        ));
    }
    Ok(value as i64)
}

// Other mark-bearing script blocks are rejected by the simple-LTR allowlist.
fn is_combining_mark(character: char) -> bool {
    matches!(
        u32::from(character),
        0x0300..=0x036F
            | 0x0483..=0x0489
            | 0x1AB0..=0x1AFF
            | 0x1DC0..=0x1DFF
            | 0x20D0..=0x20FF
            | 0x302A..=0x302F
            | 0x3099..=0x309A
            | 0xA66F
            | 0xA67C..=0xA67D
            | 0xFE20..=0xFE2F
    )
}

fn is_unsupported_format_or_emoji_component(character: char) -> bool {
    matches!(
        u32::from(character),
        0x200B..=0x200F
            | 0x202A..=0x202E
            | 0x2060..=0x206F
            | 0xFE00..=0xFE0F
            | 0xFEFF
            | 0x1F1E6..=0x1F1FF
            | 0x1F3FB..=0x1F3FF
            | 0xE0020..=0xE007F
            | 0xE0100..=0xE01EF
    )
}

// The glyph-by-glyph path intentionally admits only uncomplicated left-to-right blocks.
fn is_simple_ltr_character(character: char) -> bool {
    let codepoint = u32::from(character);
    if (character.is_control() && character != '\r')
        || matches!(
            codepoint,
            0x00AD | 0x200B..=0x200F | 0x2028..=0x202E | 0x2060..=0x206F
        )
    {
        return false;
    }
    if character.is_ascii() {
        return character == ' ' || character == '\r' || character.is_ascii_graphic();
    }
    matches!(
        codepoint,
        0x00A0..=0x02FF
            | 0x0370..=0x0482
            | 0x048A..=0x052F
            | 0x0531..=0x0588
            | 0x10A0..=0x10FF
            | 0x1E00..=0x1EFF
            | 0x1F00..=0x1FFF
            | 0x2000..=0x206F
            | 0x2070..=0x209F
            | 0x20A0..=0x20CF
            | 0x2100..=0x22FF
            | 0x2300..=0x23FF
            | 0x2460..=0x27FF
            | 0x2B00..=0x2BFF
            | 0x2C60..=0x2C7F
            | 0x2E00..=0x2E7F
            | 0x3000..=0x30FF
            | 0x3100..=0x312F
            | 0x31A0..=0x31BF
            | 0x31F0..=0x31FF
            | 0x3400..=0x4DBF
            | 0x4E00..=0x9FFF
            | 0xA640..=0xA69F
            | 0xA720..=0xA7FF
            | 0xAB30..=0xAB6F
            | 0xAC00..=0xD7A3
            | 0xF900..=0xFAFF
    )
}

fn unsupported(reason: &str) -> PotError {
    PotError::with_details(
        ErrorCode::UnsupportedFeature,
        reason,
        json!({"feature_id":"sequencer.strip.text.unicode"}),
    )
}
