//! Deterministic vector tessellation for the built-in text font.

use std::collections::HashSet;

use glam::{DVec2, DVec3};

use crate::{
    geom::{Edge, Face, Mesh, MeshError},
    model::TextObjectData,
};

const MAX_BODY_BYTES: usize = 4096;
const HERSHEY_GLYPH_COUNT: usize = 95;
const MAX_TEXT_MESH_ELEMENTS: usize = 100_000;
const FONT_CAP_HEIGHT: f64 = 21.0;
const FONT_BASELINE_OFFSET: f64 = 12.0;
const FONT_COORDINATE_MARGIN: f64 = 32.0;
const LINE_ADVANCE: f64 = 29.0;
const STROKE_WIDTH: f64 = 0.65;

/// Tessellate text in the built-in Hershey Simplex vector font into a polygon mesh.
///
/// The font accepts printable ASCII, spaces, and LF/CRLF line breaks. Unsupported
/// characters, invalid alignment names, and invalid dimensions return an error.
///
/// # Errors
///
/// Returns an error for unsupported text data, non-finite geometry, or over-budget text.
pub fn evaluate_text(data: &TextObjectData) -> Result<Mesh, MeshError> {
    if !data.size.is_finite() || data.size <= 0.0 {
        return Err(MeshError::InvalidParameter(
            "text size must be finite and positive",
        ));
    }
    if !data.extrude.is_finite() || data.extrude < 0.0 {
        return Err(MeshError::InvalidParameter(
            "text extrusion must be finite and nonnegative",
        ));
    }
    if !data.bevel_depth.is_finite() || data.bevel_depth < 0.0 {
        return Err(MeshError::InvalidParameter(
            "text bevel depth must be finite and nonnegative",
        ));
    }
    let horizontal_alignment = match data.align_x.as_str() {
        "left" => HorizontalAlignment::Left,
        "center" => HorizontalAlignment::Center,
        "right" => HorizontalAlignment::Right,
        _ => {
            return Err(MeshError::InvalidParameter(
                "text horizontal alignment must be left, center, or right",
            ));
        }
    };
    let vertical_alignment = match data.align_y.as_str() {
        "baseline" => VerticalAlignment::Baseline,
        "top" => VerticalAlignment::Top,
        "center" => VerticalAlignment::Center,
        "bottom" => VerticalAlignment::Bottom,
        _ => {
            return Err(MeshError::InvalidParameter(
                "text vertical alignment must be baseline, top, center, or bottom",
            ));
        }
    };
    if data.body.len() > MAX_BODY_BYTES {
        return Err(MeshError::InvalidParameter(
            "text body exceeds the 4096-byte limit",
        ));
    }

    let lines = split_lines(&data.body)?;
    let last_line_offset = f64::from(
        u32::try_from(lines.len().saturating_sub(1))
            .map_err(|_| MeshError::InvalidParameter("text has too many lines"))?,
    ) * LINE_ADVANCE;
    let mut glyph_cache = [None; HERSHEY_GLYPH_COUNT];
    let mut line_widths = Vec::with_capacity(lines.len());
    let mut total_segments = 0_usize;
    let mut widest_line = 0.0_f64;
    for line in &lines {
        let mut width = 0.0;
        for &character in *line {
            let glyph = cached_hershey_glyph(character, &mut glyph_cache).ok_or(
                MeshError::InvalidParameter("text body contains an unsupported character"),
            )?;
            width += glyph.advance();
            total_segments = total_segments.checked_add(glyph.segment_count).ok_or(
                MeshError::InvalidParameter("text stroke count exceeds the tessellation limit"),
            )?;
        }
        widest_line = widest_line.max(width);
        line_widths.push(width);
    }

    let unit = data.size / FONT_CAP_HEIGHT;
    let text_height_units = last_line_offset + FONT_CAP_HEIGHT;
    if !(unit.is_finite()
        && unit > 0.0
        && ((widest_line + FONT_COORDINATE_MARGIN) * unit).is_finite()
        && ((text_height_units + FONT_COORDINATE_MARGIN) * unit).is_finite())
    {
        return Err(MeshError::InvalidParameter(
            "text dimensions exceed the finite coordinate range",
        ));
    }

    let vertical_offset = match vertical_alignment {
        VerticalAlignment::Baseline => 0.0,
        VerticalAlignment::Top => -FONT_CAP_HEIGHT,
        VerticalAlignment::Center => {
            let top = FONT_CAP_HEIGHT;
            let bottom = -last_line_offset;
            -(top + bottom) / 2.0
        }
        VerticalAlignment::Bottom => last_line_offset,
    };
    let extrusion_depth = if data.extrude > 0.0 {
        data.extrude
    } else {
        data.bevel_depth
    };
    let is_solid = extrusion_depth > 0.0;
    let stroke_width = unit * STROKE_WIDTH;
    if !stroke_width.is_finite() || stroke_width <= 0.0 {
        return Err(MeshError::InvalidParameter(
            "text stroke width is outside the finite coordinate range",
        ));
    }
    let elements_per_segment = if is_solid { 26 } else { 9 };
    let expected_elements = total_segments
        .checked_mul(elements_per_segment)
        .ok_or(MeshError::InvalidParameter("text geometry size overflow"))?;
    if expected_elements > MAX_TEXT_MESH_ELEMENTS {
        return Err(MeshError::InvalidParameter(
            "text geometry exceeds the mesh element limit",
        ));
    }

    let bevel = if is_solid {
        data.bevel_depth
            .min(stroke_width * 0.45)
            .min(extrusion_depth * 0.45)
    } else {
        0.0
    };
    let mut builder = MeshBuilder::new(total_segments, is_solid);

    let mut baseline = vertical_offset;
    for (line, &line_width) in lines.iter().zip(&line_widths) {
        let horizontal_offset = match horizontal_alignment {
            HorizontalAlignment::Left => 0.0,
            HorizontalAlignment::Center => -line_width / 2.0,
            HorizontalAlignment::Right => -line_width,
        };
        let mut pen_x = 0.0;

        for &character in *line {
            let glyph = cached_hershey_glyph(character, &mut glyph_cache).ok_or(
                MeshError::InvalidParameter("text body contains an unsupported character"),
            )?;
            let glyph_origin = horizontal_offset + pen_x - glyph.left;
            let mut previous = None;
            for pair in glyph.points.as_chunks::<2>().0 {
                if pair == b" R" {
                    previous = None;
                    continue;
                }
                let coordinate = DVec3::new(
                    (glyph_origin + f64::from(pair[0]) - f64::from(b'R')) * unit,
                    (baseline + f64::from(pair[1]) - f64::from(b'R') + FONT_BASELINE_OFFSET) * unit,
                    0.0,
                );
                if let Some(start) = previous {
                    add_stroked_segment(
                        &mut builder,
                        start,
                        coordinate,
                        extrusion_depth,
                        stroke_width,
                        bevel,
                    )?;
                }
                previous = Some(coordinate);
            }
            pen_x += glyph.advance();
        }
        baseline -= LINE_ADVANCE;
    }

    builder.mesh.validate()?;
    Ok(builder.mesh)
}

#[derive(Clone, Copy)]
enum HorizontalAlignment {
    Left,
    Center,
    Right,
}

#[derive(Clone, Copy)]
enum VerticalAlignment {
    Baseline,
    Top,
    Center,
    Bottom,
}

fn split_lines(body: &str) -> Result<Vec<&[u8]>, MeshError> {
    let bytes = body.as_bytes();
    let mut lines = Vec::new();
    let mut start = 0;
    for (index, &byte) in bytes.iter().enumerate() {
        if byte == b'\n' {
            let end = if index > start && bytes[index - 1] == b'\r' {
                index - 1
            } else {
                index
            };
            if bytes[start..end].contains(&b'\r') {
                return Err(MeshError::InvalidParameter(
                    "carriage return is only supported as part of CRLF",
                ));
            }
            lines.push(&bytes[start..end]);
            start = index + 1;
        }
    }
    let final_line = &bytes[start..];
    if final_line.contains(&b'\r') {
        return Err(MeshError::InvalidParameter(
            "carriage return is only supported as part of CRLF",
        ));
    }
    lines.push(final_line);
    Ok(lines)
}

struct MeshBuilder {
    mesh: Mesh,
    edge_keys: HashSet<(u32, u32)>,
}

impl MeshBuilder {
    fn new(expected_segments: usize, solid: bool) -> Self {
        let vertices_per_segment = if solid { 8 } else { 4 };
        let edges_per_segment = if solid { 12 } else { 4 };
        let faces_per_segment = if solid { 6 } else { 1 };
        let mut mesh = Mesh::new();
        mesh.vertices
            .reserve(expected_segments.saturating_mul(vertices_per_segment));
        mesh.edges
            .reserve(expected_segments.saturating_mul(edges_per_segment));
        mesh.faces
            .reserve(expected_segments.saturating_mul(faces_per_segment));
        Self {
            mesh,
            edge_keys: HashSet::with_capacity(expected_segments.saturating_mul(edges_per_segment)),
        }
    }

    fn vertex(&mut self, coordinate: DVec3) -> Result<u32, MeshError> {
        self.mesh.insert_vertex(coordinate)
    }

    fn face(&mut self, vertices: &[u32]) -> Result<(), MeshError> {
        let face_id = self.mesh.next_id.face;
        let next_face = face_id.checked_add(1).ok_or(MeshError::IdExhausted)?;
        for index in 0..vertices.len() {
            let first = vertices[index];
            let second = vertices[(index + 1) % vertices.len()];
            let key = if first < second {
                (first, second)
            } else {
                (second, first)
            };
            if self.edge_keys.insert(key) {
                let edge_id = self.mesh.next_id.edge;
                let next_edge = edge_id.checked_add(1).ok_or(MeshError::IdExhausted)?;
                self.mesh.edges.push(Edge {
                    id: edge_id,
                    vertices: [first, second],
                });
                self.mesh.next_id.edge = next_edge;
            }
        }
        self.mesh.faces.push(Face {
            id: face_id,
            vertices: vertices.to_vec(),
            material_index: 0,
        });
        self.mesh.next_id.face = next_face;
        Ok(())
    }
}

fn add_stroked_segment(
    builder: &mut MeshBuilder,
    start: DVec3,
    end: DVec3,
    depth: f64,
    stroke_width: f64,
    bevel: f64,
) -> Result<(), MeshError> {
    let start_xy = start.truncate();
    let end_xy = end.truncate();
    let direction = end_xy - start_xy;
    let length = direction.length();
    if !length.is_finite() {
        return Err(MeshError::InvalidParameter(
            "text stroke coordinates exceed the finite range",
        ));
    }
    if length <= 0.0 {
        return Ok(());
    }
    let tangent = direction / length;
    let side = DVec2::new(-tangent.y, tangent.x);
    let half_width = stroke_width * 0.5;
    let lower = [
        start_xy + side * half_width,
        end_xy + side * half_width,
        end_xy - side * half_width,
        start_xy - side * half_width,
    ];
    if depth <= 0.0 {
        let mut ids = [0_u32; 4];
        for (index, coordinate) in lower.into_iter().enumerate() {
            ids[index] = builder.vertex(coordinate.extend(0.0))?;
        }
        return builder.face(&ids);
    }

    let bevel_along = bevel.min(length * 0.45);
    let bevel_side = bevel.min(half_width * 0.45);
    let bevel_offsets = [
        tangent * bevel_along - side * bevel_side,
        -tangent * bevel_along - side * bevel_side,
        -tangent * bevel_along + side * bevel_side,
        tangent * bevel_along + side * bevel_side,
    ];
    let mut ids = [0_u32; 8];
    for index in 0..4 {
        ids[index] = builder.vertex(lower[index].extend(0.0))?;
        ids[index + 4] = builder.vertex((lower[index] + bevel_offsets[index]).extend(depth))?;
    }
    builder.face(&[ids[0], ids[3], ids[2], ids[1]])?;
    builder.face(&[ids[4], ids[5], ids[6], ids[7]])?;
    builder.face(&[ids[0], ids[1], ids[5], ids[4]])?;
    builder.face(&[ids[1], ids[2], ids[6], ids[5]])?;
    builder.face(&[ids[2], ids[3], ids[7], ids[6]])?;
    builder.face(&[ids[3], ids[0], ids[4], ids[7]])?;
    Ok(())
}

// Roman Simplex records from the public-domain Hershey vector-font collection.
// The JHF glyph data is permissive-use-and-redistribution data; records map in
// printable ASCII order from space (32) through tilde (126).
const HERSHEY_SIMPLEX_JHF: &str = r"  699  1JZ
  714  9MWRFRT RRYQZR[SZRY
  717  6JZNFNM RVFVM
  733 12H]SBLb RYBRb RLOZO RKUYU
  719 27H\PBP_ RTBT_ RYIWGTFPFMGKIKKLMMNOOUQWRXSYUYXWZT[P[MZKX
 2271 32F^[FI[ RNFPHPJOLMMKMIKIIJGLFNFPGSHVHYG[F RWTUUTWTYV[X[ZZ[X[VYTWT
  734 35E_\O\N[MZMYNXPVUTXRZP[L[JZIYHWHUISJRQNRMSKSIRGPFNGMIMKNNPQUXWZY[[[\Z\Y
  731  8MWRHQGRFSGSIRKQL
  721 11KYVBTDRGPKOPOTPYR]T`Vb
  722 11KYNBPDRGTKUPUTTYR]P`Nb
 2219  9JZRFRR RMIWO RWIMO
  725  6E_RIR[ RIR[R
  711  9MWSZR[QZRYSZS\R^Q_
  724  3E_IR[R
  710  6MWRYQZR[SZRY
  720  3G][BIb
  700 18H\QFNGLJKOKRLWNZQ[S[VZXWYRYOXJVGSFQF
  701  5H\NJPISFS[
  702 15H\LKLJMHNGPFTFVGWHXJXLWNUQK[Y[
  703 16H\MFXFRNUNWOXPYSYUXXVZS[P[MZLYKW
  704  7H\UFKTZT RUFU[
  705 18H\WFMFLOMNPMSMVNXPYSYUXXVZS[P[MZLYKW
  706 24H\XIWGTFRFOGMJLOLTMXOZR[S[VZXXYUYTXQVOSNRNOOMQLT
  707  6H\YFO[ RKFYF
  708 30H\PFMGLILKMMONSOVPXRYTYWXYWZT[P[MZLYKWKTLRNPQOUNWMXKXIWGTFPF
  709 24H\XMWPURRSQSNRLPKMKLLINGQFRFUGWIXMXRWWUZR[P[MZLX
  712 12MWRMQNROSNRM RRYQZR[SZRY
  713 15MWRMQNROSNRM RSZR[QZRYSZS\R^Q_
 2241  4F^ZIJRZ[
  726  6E_IO[O RIU[U
 2242  4F^JIZRJ[
  715 21I[LKLJMHNGPFTFVGWHXJXLWNVORQRT RRYQZR[SZRY
 2273 56E`WNVLTKQKOLNMMPMSNUPVSVUUVS RQKOMNPNSOUPV RWKVSVUXVZV\T]Q]O\L[JYHWGTFQFNGLHJJILHOHRIUJWLYNZQ[T[WZYYZX RXKWSWUXV
  501  9I[RFJ[ RRFZ[ RMTWT
  502 24G\KFK[ RKFTFWGXHYJYLXNWOTP RKPTPWQXRYTYWXYWZT[K[
  503 19H]ZKYIWGUFQFOGMILKKNKSLVMXOZQ[U[WZYXZV
  504 16G\KFK[ RKFRFUGWIXKYNYSXVWXUZR[K[
  505 12H[LFL[ RLFYF RLPTP RL[Y[
  506  9HZLFL[ RLFYF RLPTP
  507 23H]ZKYIWGUFQFOGMILKKNKSLVMXOZQ[U[WZYXZVZS RUSZS
  508  9G]KFK[ RYFY[ RKPYP
  509  3NVRFR[
  510 11JZVFVVUYTZR[P[NZMYLVLT
  511  9G\KFK[ RYFKT RPOY[
  512  6HYLFL[ RL[X[
  513 12F^JFJ[ RJFR[ RZFR[ RZFZ[
  514  9G]KFK[ RKFY[ RYFY[
  515 22G]PFNGLIKKJNJSKVLXNZP[T[VZXXYVZSZNYKXIVGTFPF
  516 14G\KFK[ RKFTFWGXHYJYMXOWPTQKQ
  517 25G]PFNGLIKKJNJSKVLXNZP[T[VZXXYVZSZNYKXIVGTFPF RSWY]
  518 17G\KFK[ RKFTFWGXHYJYLXNWOTPKP RRPY[
  519 21H\YIWGTFPFMGKIKKLMMNOOUQWRXSYUYXWZT[P[MZKX
  520  6JZRFR[ RKFYF
  521 11G]KFKULXNZQ[S[VZXXYUYF
  522  6I[JFR[ RZFR[
  523 12F^HFM[ RRFM[ RRFW[ R\FW[
  524  6H\KFY[ RYFK[
  525  7I[JFRPR[ RZFRP
  526  9H\YFK[ RKFYF RK[Y[
 2223 12KYOBOb RPBPb ROBVB RObVb
  804  3KYKFY^
 2224 12KYTBTb RUBUb RNBUB RNbUb
 2262 11JZPLRITL RMORJWO RRJR[
  999  3JZJ]Z]
  730  8MWSFRGQIQKRLSKRJ
  601 18I\XMX[ RXPVNTMQMONMPLSLUMXOZQ[T[VZXX
  602 18H[LFL[ RLPNNPMSMUNWPXSXUWXUZS[P[NZLX
  603 15I[XPVNTMQMONMPLSLUMXOZQ[T[VZXX
  604 18I\XFX[ RXPVNTMQMONMPLSLUMXOZQ[T[VZXX
  605 18I[LSXSXQWOVNTMQMONMPLSLUMXOZQ[T[VZXX
  606  9MYWFUFSGRJR[ ROMVM
  607 23I\XMX]W`VaTbQbOa RXPVNTMQMONMPLSLUMXOZQ[T[VZXX
  608 11I\MFM[ RMQPNRMUMWNXQX[
  609  9NVQFRGSFREQF RRMR[
  610 12MWRFSGTFSERF RSMS^RaPbNb
  611  9IZMFM[ RWMMW RQSX[
  612  3NVRFR[
  613 19CaGMG[ RGQJNLMOMQNRQR[ RRQUNWMZM\N]Q][
  614 11I\MMM[ RMQPNRMUMWNXQX[
  615 18I\QMONMPLSLUMXOZQ[T[VZXXYUYSXPVNTMQM
  616 18H[LMLb RLPNNPMSMUNWPXSXUWXUZS[P[NZLX
  617 18I\XMXb RXPVNTMQMONMPLSLUMXOZQ[T[VZXX
  618  9KXOMO[ ROSPPRNTMWM
  619 18J[XPWNTMQMNNMPNRPSUTWUXWXXWZT[Q[NZMX
  620  9MYRFRWSZU[W[ ROMVM
  621 11I\MMMWNZP[S[UZXW RXMX[
  622  6JZLMR[ RXMR[
  623 12G]JMN[ RRMN[ RRMV[ RZMV[
  624  6J[MMX[ RXMM[
  625 10JZLMR[ RXMR[P_NaLbKb
  626  9J[XMM[ RMMXM RM[X[
 2225 40KYTBRCQDPFPHQJRKSMSOQQ RRCQEQGRISJTLTNSPORSTTVTXSZR[Q]Q_Ra RQSSUSWRYQZP\P^Q`RaTb
  723  3NVRBRb
 2226 40KYPBRCSDTFTHSJRKQMQOSQ RRCSESGRIQJPLPNQPURQTPVPXQZR[S]S_Ra RSSQUQWRYSZT\T^S`RaPb
 2246 24F^IUISJPLONOPPTSVTXTZS[Q RISJQLPNPPQTTVUXUZT[Q[O
  718 14KYQFOGNINKOMQNSNUMVKVIUGSFQF";

#[derive(Clone, Copy)]
struct HersheyGlyph {
    left: f64,
    right: f64,
    points: &'static [u8],
    segment_count: usize,
}

impl HersheyGlyph {
    fn advance(self) -> f64 {
        self.right - self.left
    }
}

fn cached_hershey_glyph(
    character: u8,
    cache: &mut [Option<HersheyGlyph>; HERSHEY_GLYPH_COUNT],
) -> Option<HersheyGlyph> {
    let record_index = usize::from(character.checked_sub(b' ')?);
    let cached = cache.get_mut(record_index)?;
    if let Some(glyph) = cached {
        return Some(*glyph);
    }
    let glyph = parse_hershey_glyph(record_index)?;
    *cached = Some(glyph);
    Some(glyph)
}

fn parse_hershey_glyph(record_index: usize) -> Option<HersheyGlyph> {
    let line = HERSHEY_SIMPLEX_JHF.lines().nth(record_index)?;
    let bytes = line.as_bytes();
    let pair_count = line.get(5..8)?.trim().parse::<usize>().ok()?;
    let points = bytes.get(10..)?;
    if pair_count == 0 || points.len() != pair_count.checked_sub(1)?.checked_mul(2)? {
        return None;
    }
    let left = f64::from(*bytes.get(8)?) - f64::from(b'R');
    let right = f64::from(*bytes.get(9)?) - f64::from(b'R');
    let mut segment_count = 0_usize;
    let mut has_previous = false;
    for pair in points.as_chunks::<2>().0 {
        if pair == b" R" {
            has_previous = false;
        } else if has_previous {
            segment_count += 1;
        } else {
            has_previous = true;
        }
    }
    Some(HersheyGlyph {
        left,
        right,
        points,
        segment_count,
    })
}
