use ttf_parser::{Face, GlyphId, OutlineBuilder};

use crate::{
    geom::{Mesh, MeshError, curve::evaluate_curve},
    model::{
        CurveData, CurveDimensions, CurveFillMode, CurveFillSolver, CurveHandleType, CurvePoint,
        CurveSpline, CurveSplineType, TextCharacterFormat, TextObjectData,
    },
};

const MAX_FONT_POINTS: usize = 65_536;

#[derive(Clone, Copy)]
pub(crate) enum FontVariantSource<'a> {
    Inherit,
    Builtin,
    Missing,
    File(&'a [u8]),
}

#[derive(Clone, Copy)]
pub(crate) struct FontSources<'a> {
    pub base: Option<&'a [u8]>,
    pub bold: FontVariantSource<'a>,
    pub italic: FontVariantSource<'a>,
    pub bold_italic: FontVariantSource<'a>,
}

struct FontFace<'a> {
    face: Face<'a>,
    scale: f32,
    em_ratio: f32,
    ascend_ratio: f32,
}

impl<'a> FontFace<'a> {
    fn parse(bytes: &'a [u8]) -> Result<Self, MeshError> {
        let face = Face::parse(bytes, 0).map_err(|_| MeshError::UnsupportedFeature {
            feature_id: "text.font.external.format",
            reason: "the external font is not a supported TrueType/OpenType outline font",
        })?;
        if face.is_variable() {
            return Err(MeshError::UnsupportedFeature {
                feature_id: "text.font.external.variable",
                reason: "variable external fonts are not supported",
            });
        }

        if !face.tables().cmap.is_some_and(|cmap| {
            cmap.subtables
                .into_iter()
                .any(|subtable| subtable.is_unicode())
        }) {
            return Err(MeshError::UnsupportedFeature {
                feature_id: "text.font.external.charmap",
                reason: "external font files must provide a Unicode character map",
            });
        }
        let bbox = face.global_bounding_box();
        let bbox_height = i32::from(bbox.y_max) - i32::from(bbox.y_min);
        let ascender = face.ascender();
        let descender = face.descender();
        let vertical_span = i32::from(ascender) - i32::from(descender);
        let complete = ascender != 0 && descender != 0 && ascender != descender;
        let ascend_ratio = if complete {
            f32::from(ascender) / vertical_span as f32
        } else {
            0.8
        };
        let scale = if bbox_height != 0 {
            (1.0 / f64::from(bbox_height)) as f32
        } else {
            0.001
        };
        let em_ratio = if complete {
            if bbox_height != 0 {
                vertical_span as f32 / bbox_height as f32
            } else {
                0.0
            }
        } else {
            1.0
        };
        Ok(Self {
            face,
            scale,
            em_ratio,
            ascend_ratio,
        })
    }

    fn advance(&self, glyph: GlyphId) -> f32 {
        self.face
            .glyph_hor_advance(glyph)
            .map_or(0.0, |advance| f32::from(advance) * self.scale)
    }

    fn glyph(&self, character: char) -> Option<GlyphId> {
        self.face
            .glyph_index(character)
            .filter(|glyph| glyph.0 != 0)
    }

    fn outline(&self, glyph: GlyphId, resolution: u32) -> Result<Vec<CurveSpline>, MeshError> {
        let mut builder = FontOutlineBuilder::new(self.scale, resolution);
        let _ = self.face.outline_glyph(glyph, &mut builder);
        if builder.failed {
            return Err(MeshError::InvalidParameter(
                "font glyph outline exceeds the curve tessellation limit",
            ));
        }
        Ok(builder.splines)
    }
}

struct LoadedFont<'a> {
    base: Option<FontFace<'a>>,
    bold: VariantFont<'a>,
    italic: VariantFont<'a>,
    bold_italic: VariantFont<'a>,
}

enum VariantFont<'a> {
    Inherit,
    Builtin,
    Missing,
    Face(Box<FontFace<'a>>),
}

impl<'a> LoadedFont<'a> {
    fn new(sources: FontSources<'a>) -> Result<Self, MeshError> {
        Ok(Self {
            base: sources.base.map(FontFace::parse).transpose()?,
            bold: Self::load_variant(sources.bold)?,
            italic: Self::load_variant(sources.italic)?,
            bold_italic: Self::load_variant(sources.bold_italic)?,
        })
    }

    fn load_variant(source: FontVariantSource<'a>) -> Result<VariantFont<'a>, MeshError> {
        match source {
            FontVariantSource::Inherit => Ok(VariantFont::Inherit),
            FontVariantSource::Builtin => Ok(VariantFont::Builtin),
            FontVariantSource::Missing => Ok(VariantFont::Missing),
            FontVariantSource::File(bytes) => {
                FontFace::parse(bytes).map(|face| VariantFont::Face(Box::new(face)))
            }
        }
    }

    fn primary_metrics(&self) -> (f32, f32) {
        self.base.as_ref().map_or((0.8, 1.0), |face| {
            (face.ascend_ratio * face.em_ratio, face.em_ratio)
        })
    }

    fn face_for(&self, format: &TextCharacterFormat) -> Result<Option<&FontFace<'a>>, MeshError> {
        let variant = match (format.use_bold, format.use_italic) {
            (true, true) => Some(&self.bold_italic),
            (true, false) => Some(&self.bold),
            (false, true) => Some(&self.italic),
            (false, false) => None,
        };
        let Some(variant) = variant else {
            return Ok(self.base.as_ref());
        };
        match variant {
            VariantFont::Inherit => Ok(self.base.as_ref()),
            VariantFont::Missing => Ok(None),
            VariantFont::Face(face) => Ok(Some(face)),
            VariantFont::Builtin => Err(MeshError::UnsupportedFeature {
                feature_id: "text.font.external.style_builtin_variant",
                reason: "Blender's built-in bold or italic font variant cannot be parsed as an external font",
            }),
        }
    }
}

#[derive(Clone, Copy)]
struct Placement {
    char_index: usize,
    x: f32,
    y: f32,
    glyph: Option<GlyphId>,
    small_caps: bool,
}

pub(crate) fn evaluate_external_text(
    data: &TextObjectData,
    sources: FontSources<'_>,
) -> Result<Mesh, MeshError> {
    if data.bevel_depth as f32 > 0.0 {
        return Err(MeshError::UnsupportedFeature {
            feature_id: "text.font.external.bevel",
            reason: "external font outlines do not yet support text bevel geometry",
        });
    }

    let size = data.size as f32;
    let character_spacing = data.character_spacing as f32;
    let word_spacing = data.word_spacing as f32;
    let line_spacing = data.line_spacing as f32;
    let shear = data.shear as f32;
    let offset_x = data.offset_x as f32;
    let offset_y = data.offset_y as f32;
    let small_caps_scale = data.small_caps_scale as f32;
    if !(size.is_finite()
        && size > 0.0
        && character_spacing.is_finite()
        && word_spacing.is_finite()
        && line_spacing.is_finite()
        && shear.is_finite()
        && offset_x.is_finite()
        && offset_y.is_finite()
        && small_caps_scale.is_finite())
    {
        return Err(MeshError::InvalidParameter(
            "external text settings must be finite",
        ));
    }
    let font = LoadedFont::new(sources)?;
    let (ascent, em_ratio) = font.primary_metrics();
    let mut lines = Vec::with_capacity(data.body.split('\n').count());
    let mut char_index = 0_usize;
    let mut widest_line = 0.0_f32;
    let mut total_widths = Vec::with_capacity(lines.capacity());

    for line in data.body.split('\n') {
        let mut placements = Vec::with_capacity(line.chars().count());
        let mut pen_x = 0.0_f32;
        for character in line.chars() {
            let format = character_format(data, char_index);
            validate_character_format(format)?;
            if character == '\t' {
                let tab = 2.0 * f32::midpoint(pen_x, 0.01).ceil();
                pen_x = tab;
                char_index = char_index.saturating_add(1);
                continue;
            }
            if character == '\0' {
                char_index = char_index.saturating_add(1);
                continue;
            }

            let (glyph_character, small_caps) =
                small_caps_character(character, format.use_small_caps);
            let glyph_character = normalize_control(glyph_character);
            let face = font.face_for(format)?;
            let glyph = face.and_then(|font| font.glyph(glyph_character));
            if glyph.is_none() && glyph_character != ' ' {
                return Err(MeshError::UnsupportedFeature {
                    feature_id: "text.font.external.missing_glyph",
                    reason: "the external font does not contain the requested character",
                });
            }
            let advance = if let (Some(face), Some(glyph)) = (face, glyph) {
                let advance = face.advance(glyph);
                if glyph_character != ' ' && advance == 0.0 {
                    return Err(MeshError::UnsupportedFeature {
                        feature_id: "text.font.external.combining_mark",
                        reason: "external font glyphs with zero advance are not supported",
                    });
                }
                advance
            } else {
                0.5 * em_ratio
            } * if small_caps { small_caps_scale } else { 1.0 };
            let word_factor = if character == ' ' { word_spacing } else { 1.0 };
            let kern_factor = 1.0 + format.kerning as f32 / 40.0;
            let extra = if character_spacing < 1.0 {
                (character_spacing - 1.0) * advance
            } else {
                0.5 * character_spacing - 0.5
            };
            placements.push(Placement {
                char_index,
                x: pen_x,
                y: 0.0,
                glyph,
                small_caps,
            });
            pen_x += advance * word_factor * kern_factor + extra;
            char_index = char_index.saturating_add(1);
        }
        let line_width = if line.is_empty() || character_spacing < 1.0 {
            pen_x
        } else {
            pen_x - (0.5 * character_spacing - 0.5)
        };
        widest_line = widest_line.max(line_width);
        total_widths.push(line_width);
        lines.push(placements);
        char_index = char_index.saturating_add(1);
    }
    let last_line = lines.len().saturating_sub(1) as f32;
    let vertical_offset = match data.align_y.as_str() {
        "baseline" => 0.0,
        "bottom_exact" => {
            return Err(MeshError::UnsupportedFeature {
                feature_id: "text.font.external.align_y.bottom",
                reason: "exact bottom text alignment is not supported for external fonts",
            });
        }
        "top" => -ascent,
        "center" => f32::midpoint(em_ratio, last_line * line_spacing) - ascent,
        "bottom" => last_line * line_spacing,
        _ => {
            return Err(MeshError::InvalidParameter(
                "text vertical alignment must be baseline, top, center, or bottom",
            ));
        }
    };
    if !matches!(data.align_x.as_str(), "left" | "center" | "right") {
        return Err(MeshError::InvalidParameter(
            "text horizontal alignment must be left, center, or right",
        ));
    }
    if !(widest_line.is_finite() && vertical_offset.is_finite()) {
        return Err(MeshError::InvalidParameter(
            "external text dimensions exceed the finite coordinate range",
        ));
    }

    let mut splines = Vec::new();
    let mut control_points = 0_usize;
    let mut line_baseline = vertical_offset + offset_y / size;
    for (line, width) in lines.iter().zip(total_widths) {
        let align_x = match data.align_x.as_str() {
            "left" => 0.0,
            "center" => -width / 2.0,
            "right" => -width,
            _ => {
                return Err(MeshError::InvalidParameter(
                    "text horizontal alignment must be left, center, or right",
                ));
            }
        };
        for placement in line {
            let Some(glyph) = placement.glyph else {
                continue;
            };
            let format = character_format(data, placement.char_index);
            let Some(face) = font.face_for(format)? else {
                return Err(MeshError::UnsupportedFeature {
                    feature_id: "text.font.external.missing_glyph",
                    reason: "the external font variant is unavailable for the requested character",
                });
            };
            let mut outlines = face.outline(glyph, data.resolution_u)?;
            reserve_font_points(
                &mut control_points,
                outlines.iter().map(|spline| spline.points.len()).sum(),
            )?;
            let small_caps = if placement.small_caps {
                small_caps_scale
            } else {
                1.0
            };
            let x_offset = align_x + placement.x + offset_x / size;
            let y_offset = line_baseline + placement.y;
            for spline in &mut outlines {
                transform_spline(spline, shear, small_caps, x_offset, y_offset, size);
            }
            splines.extend(outlines);
        }
        line_baseline -= line_spacing;
    }

    let fill_mode = data.fill_mode;
    if fill_mode == CurveFillMode::None && data.extrude > 0.0 {
        return Err(MeshError::UnsupportedFeature {
            feature_id: "text.font.external.extrusion_without_fill",
            reason: "external text extrusion without a 2D fill is not supported",
        });
    }
    if fill_mode != CurveFillMode::None && data.fill_solver != CurveFillSolver::Cdt {
        return Err(MeshError::UnsupportedFeature {
            feature_id: "text.font.external.fill_solver",
            reason: "external font fill supports Blender's CDT solver only",
        });
    }
    let curve = CurveData {
        splines,
        dimensions: CurveDimensions::TwoD,
        fill_mode,
        fill_solver: data.fill_solver,
        fill_rule: data.fill_rule,
        extrude: f64::from(data.extrude as f32),
        ..CurveData::default()
    };
    evaluate_curve(&curve)
}

fn character_format(data: &TextObjectData, index: usize) -> &TextCharacterFormat {
    data.character_formats
        .get(index)
        .unwrap_or_else(|| default_character_format())
}

fn default_character_format() -> &'static TextCharacterFormat {
    static DEFAULT: TextCharacterFormat = TextCharacterFormat {
        use_bold: false,
        use_italic: false,
        use_underline: false,
        use_small_caps: false,
        material_index: 0,
        kerning: 0.0,
    };
    &DEFAULT
}

fn validate_character_format(format: &TextCharacterFormat) -> Result<(), MeshError> {
    if format.use_underline {
        return Err(MeshError::UnsupportedFeature {
            feature_id: "text.font.external.underline",
            reason: "external text font underlines are not supported",
        });
    }
    if format.material_index != 0 {
        return Err(MeshError::UnsupportedFeature {
            feature_id: "text.font.external.character_material",
            reason: "per-character text material assignments are not supported",
        });
    }
    if !format.kerning.is_finite() {
        return Err(MeshError::InvalidParameter(
            "text character kerning must be finite",
        ));
    }
    Ok(())
}

fn small_caps_character(character: char, enabled: bool) -> (char, bool) {
    if !enabled {
        return (character, false);
    }
    let mut uppercase = character.to_uppercase();
    let Some(first) = uppercase.next() else {
        return (character, false);
    };
    if uppercase.next().is_some() {
        (character, false)
    } else {
        (first, first != character)
    }
}

fn normalize_control(character: char) -> char {
    if (character as u32) < 32 && character != '\u{10}' && character != '\u{13}' {
        ' '
    } else {
        character
    }
}

fn reserve_font_points(total: &mut usize, additional: usize) -> Result<(), MeshError> {
    *total = total
        .checked_add(additional)
        .ok_or(MeshError::InvalidParameter(
            "text font outline point count overflow",
        ))?;
    if *total > MAX_FONT_POINTS {
        return Err(MeshError::InvalidParameter(
            "text font outlines exceed the curve control-point limit",
        ));
    }
    Ok(())
}

fn transform_spline(
    spline: &mut CurveSpline,
    shear: f32,
    small_caps: f32,
    x_offset: f32,
    y_offset: f32,
    size: f32,
) {
    for point in &mut spline.points {
        for coordinates in [
            &mut point.handle_left,
            &mut point.co,
            &mut point.handle_right,
        ] {
            let y = coordinates[1] as f32;
            let x = coordinates[0] as f32 + shear * y;
            coordinates[0] = f64::from((x * small_caps + x_offset) * size);
            coordinates[1] = f64::from((y * small_caps + y_offset) * size);
        }
    }
}

struct Contour {
    start: [f32; 2],
    current: [f32; 2],
    points: Vec<CurvePoint>,
}

struct FontOutlineBuilder {
    scale: f32,
    resolution: u32,
    point_count: usize,
    failed: bool,
    current: Option<Contour>,
    splines: Vec<CurveSpline>,
}

impl FontOutlineBuilder {
    fn new(scale: f32, resolution: u32) -> Self {
        Self {
            scale,
            resolution,
            point_count: 0,
            failed: false,
            current: None,
            splines: Vec::new(),
        }
    }

    fn point(&self, point: [f32; 2]) -> [f64; 3] {
        [
            f64::from(point[0] * self.scale),
            f64::from(point[1] * self.scale),
            0.0,
        ]
    }

    fn line_handle(&self, point: [f32; 2], neighbor: [f32; 2]) -> [f64; 3] {
        [
            f64::from(point[0] * self.scale - (point[0] - neighbor[0]) * self.scale / 3.0),
            f64::from(point[1] * self.scale - (point[1] - neighbor[1]) * self.scale / 3.0),
            0.0,
        ]
    }

    fn append_endpoint(
        &mut self,
        endpoint: [f32; 2],
        right_handle: [f64; 3],
        left_handle: [f64; 3],
        right_type: CurveHandleType,
        left_type: CurveHandleType,
    ) {
        if self.current.is_none() {
            self.failed = true;
            return;
        }
        let co = self.point(endpoint);
        let Some(contour) = self.current.as_mut() else {
            self.failed = true;
            return;
        };
        if endpoint[0].to_bits() == contour.start[0].to_bits()
            && endpoint[1].to_bits() == contour.start[1].to_bits()
            && contour.points.len() > 1
        {
            if let Some(last) = contour.points.last_mut() {
                last.handle_right = right_handle;
                last.handle_right_type = Some(right_type);
            }
            if let Some(first) = contour.points.first_mut() {
                first.handle_left = left_handle;
                first.handle_left_type = Some(left_type);
            }
            contour.current = endpoint;
            return;
        }
        if self.point_count >= MAX_FONT_POINTS {
            self.failed = true;
            return;
        }
        if let Some(last) = contour.points.last_mut() {
            last.handle_right = right_handle;
            last.handle_right_type = Some(right_type);
        }
        contour.points.push(CurvePoint {
            co,
            handle_left: left_handle,
            handle_right: co,
            handle_type: CurveHandleType::Free,
            handle_left_type: Some(left_type),
            handle_right_type: Some(CurveHandleType::Free),
            weight: 1.0,
            radius: 1.0,
            tilt: 0.0,
        });
        contour.current = endpoint;
        self.point_count += 1;
    }
    fn close_contour(&mut self) {
        let Some(contour) = self.current.take() else {
            self.failed = true;
            return;
        };
        if contour.points.len() >= 3 {
            self.splines.push(CurveSpline {
                spline_type: CurveSplineType::Bezier,
                points: contour.points,
                order: 3,
                cyclic: true,
                resolution: self.resolution,
                use_endpoint: false,
                use_bezier: false,
                custom_knots: false,
            });
        }
    }
}

impl OutlineBuilder for FontOutlineBuilder {
    fn move_to(&mut self, x: f32, y: f32) {
        if self.current.is_some() {
            self.failed = true;
            return;
        }
        if self.point_count >= MAX_FONT_POINTS {
            self.failed = true;
            return;
        }
        let start = [x, y];
        let co = self.point(start);
        self.current = Some(Contour {
            start,
            current: start,
            points: vec![CurvePoint {
                co,
                handle_left: co,
                handle_right: co,
                handle_type: CurveHandleType::Free,
                handle_left_type: Some(CurveHandleType::Free),
                handle_right_type: Some(CurveHandleType::Free),
                weight: 1.0,
                radius: 1.0,
                tilt: 0.0,
            }],
        });
        self.point_count += 1;
    }

    fn line_to(&mut self, x: f32, y: f32) {
        let Some(contour) = self.current.as_ref() else {
            self.failed = true;
            return;
        };
        let start = contour.current;
        let endpoint = [x, y];
        let right = self.line_handle(start, endpoint);
        let left = self.line_handle(endpoint, start);
        self.append_endpoint(
            endpoint,
            right,
            left,
            CurveHandleType::Vector,
            CurveHandleType::Vector,
        );
    }

    fn quad_to(&mut self, x1: f32, y1: f32, x: f32, y: f32) {
        let Some(contour) = self.current.as_ref() else {
            self.failed = true;
            return;
        };
        let start = contour.current;
        let control = [x1, y1];
        let endpoint = [x, y];
        let right = [
            f64::from((start[0] + 2.0 * control[0]) * self.scale / 3.0),
            f64::from((start[1] + 2.0 * control[1]) * self.scale / 3.0),
            0.0,
        ];
        let left = [
            f64::from((endpoint[0] + 2.0 * control[0]) * self.scale / 3.0),
            f64::from((endpoint[1] + 2.0 * control[1]) * self.scale / 3.0),
            0.0,
        ];
        self.append_endpoint(
            endpoint,
            right,
            left,
            CurveHandleType::Free,
            CurveHandleType::Free,
        );
    }

    fn curve_to(&mut self, x1: f32, y1: f32, x2: f32, y2: f32, x: f32, y: f32) {
        let Some(contour) = self.current.as_ref() else {
            self.failed = true;
            return;
        };
        let endpoint = [x, y];
        let right = [f64::from(x1 * self.scale), f64::from(y1 * self.scale), 0.0];
        let left = [f64::from(x2 * self.scale), f64::from(y2 * self.scale), 0.0];
        let _ = contour;
        self.append_endpoint(
            endpoint,
            right,
            left,
            CurveHandleType::Free,
            CurveHandleType::Free,
        );
    }

    fn close(&mut self) {
        self.close_contour();
    }
}

#[cfg(test)]
mod tests {
    #![expect(
        clippy::unwrap_used,
        clippy::expect_used,
        reason = "the regression fixture is a committed, valid Lato font"
    )]
    use super::FontFace;
    use ttf_parser::Face;

    #[test]
    fn free_type_metrics_honor_use_typo_metrics() {
        let mut font =
            include_bytes!("../../../../tests/fixtures/text_external_font/Lato-Regular.ttf")
                .to_vec();
        let (original_hhea, bbox_height) = {
            let original_face = Face::parse(&font, 0).unwrap();
            let hhea = original_face.tables().hhea;
            let bbox = original_face.global_bounding_box();
            (hhea, i32::from(bbox.y_max) - i32::from(bbox.y_min))
        };
        let table_count = usize::from(u16::from_be_bytes([font[4], font[5]]));
        let os2_offset = (0..table_count)
            .find_map(|index| {
                let record = 12 + index * 16;
                (&font[record..record + 4] == b"OS/2").then(|| {
                    u32::from_be_bytes(font[record + 8..record + 12].try_into().unwrap()) as usize
                })
            })
            .expect("Lato has an OS/2 table");

        let typo_ascender = 1000_i16;
        let typo_descender = -1000_i16;
        let selection_offset = os2_offset + 62;
        let selection = u16::from_be_bytes(
            font[selection_offset..selection_offset + 2]
                .try_into()
                .unwrap(),
        );
        font[selection_offset..selection_offset + 2]
            .copy_from_slice(&(selection | (1 << 7)).to_be_bytes());
        font[os2_offset + 68..os2_offset + 70].copy_from_slice(&typo_ascender.to_be_bytes());
        font[os2_offset + 70..os2_offset + 72].copy_from_slice(&typo_descender.to_be_bytes());

        let parsed_face = Face::parse(&font, 0).unwrap();
        let hhea = parsed_face.tables().hhea;
        assert_eq!(hhea.ascender, original_hhea.ascender);
        assert_eq!(hhea.descender, original_hhea.descender);
        let os2 = parsed_face.tables().os2.unwrap();
        assert!(os2.use_typographic_metrics());
        assert_eq!(os2.typographic_ascender(), typo_ascender);
        assert_eq!(os2.typographic_descender(), typo_descender);

        let hhea_vertical_span = i32::from(hhea.ascender) - i32::from(hhea.descender);
        let hhea_ascend_ratio = f32::from(hhea.ascender) / hhea_vertical_span as f32;
        let typo_vertical_span = i32::from(typo_ascender) - i32::from(typo_descender);
        let typo_ascend_ratio = f32::from(typo_ascender) / typo_vertical_span as f32;
        assert_ne!(hhea_ascend_ratio, typo_ascend_ratio);

        let font_face = FontFace::parse(&font).unwrap();
        assert_eq!(parsed_face.ascender(), typo_ascender);
        assert_eq!(parsed_face.descender(), typo_descender);
        assert_eq!(font_face.ascend_ratio, typo_ascend_ratio);
        assert_eq!(
            font_face.em_ratio,
            typo_vertical_span as f32 / bbox_height as f32
        );
    }
}
