use std::{collections::BTreeMap, fmt::Write as _, path::Path};

use serde_json::json;

use crate::{
    error::{ErrorCode, PotError, Result},
    eval::Snapshot,
    model::{DataBlock, Id, Material, Node, SceneDoc, Transform},
};

use super::{
    ImportedGraph, import_error, render_visible_nodes,
    vector::{self, Bounds, Projection},
};

const MESH_STROKE_WIDTH: f64 = 0.55;
const CUBIC_STEPS: u32 = 16;

struct StrokeDraw {
    points: Vec<[f64; 2]>,
    width: f64,
    color: [f64; 4],
    cyclic: bool,
}

fn view_box(bounds: &Bounds) -> Result<[f64; 4]> {
    if !bounds.min[0].is_finite() {
        return Ok([-0.5, -0.5, 1.0, 1.0]);
    }
    let mut width = bounds.max[0] - bounds.min[0];
    let mut height = bounds.max[1] - bounds.min[1];
    let mut x = bounds.min[0];
    let mut y = bounds.min[1];
    if width == 0.0 {
        width = 1.0;
        x -= 0.5;
    }
    if height == 0.0 {
        height = 1.0;
        y -= 0.5;
    }
    if !x.is_finite()
        || !y.is_finite()
        || !width.is_finite()
        || !height.is_finite()
        || width <= 0.0
        || height <= 0.0
    {
        return Err(PotError::new(
            ErrorCode::EvaluationFailed,
            "SVG viewBox cannot represent evaluated geometry",
        ));
    }
    Ok([x, y, width, height])
}

/// Export evaluated strokes and mesh wireframes as deterministic SVG.
///
/// `view` accepts `front`, `back`, `right`, `left`, `top`, `bottom`, `iso`, or
/// `isometric`. Coordinates remain in the projected scene plane.
pub fn export(doc: &SceneDoc, snapshot: &Snapshot, view: &str) -> Result<Vec<u8>> {
    let projection = parse_projection(view)?;
    let visible_nodes = render_visible_nodes(doc, snapshot)?;
    let mut bounds = Bounds::new();
    let mut mesh_lines = Vec::<[[f64; 2]; 2]>::new();
    vector::for_each_projected_mesh_edge(doc, snapshot, projection, |line| {
        for point in line {
            bounds.include(
                point,
                MESH_STROKE_WIDTH * 0.5,
                "SVG geometry bounds are not finite",
            )?;
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
        let width = (max_radius * 2.0).max(0.5);
        let color = stroke.color.map(|channel| channel.clamp(0.0, 1.0));
        for point in &points {
            bounds.include(*point, width * 0.5, "SVG geometry bounds are not finite")?;
        }
        strokes.push(StrokeDraw {
            points,
            width,
            color,
            cyclic: stroke.cyclic,
        });
        Ok(())
    })?;

    let view_box = view_box(&bounds)?;
    let mut svg = String::from("<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"");
    write_numbers(&mut svg, view_box);
    svg.push_str("\">\n");
    for [from, to] in mesh_lines {
        let _ = writeln!(
            svg,
            "  <path d=\"M {:.17e} {:.17e} L {:.17e} {:.17e}\" fill=\"none\" stroke=\"#292929\" stroke-width=\"{MESH_STROKE_WIDTH:.17e}\"/>",
            from[0], from[1], to[0], to[1],
        );
    }
    for stroke in strokes {
        let [red, green, blue] = color_channels(stroke.color);
        let _ = write!(
            svg,
            "  <path d=\"M {:.17e} {:.17e}",
            stroke.points[0][0], stroke.points[0][1],
        );
        for point in stroke.points.iter().skip(1) {
            let _ = write!(svg, " L {:.17e} {:.17e}", point[0], point[1]);
        }
        if stroke.cyclic {
            svg.push_str(" Z");
        }
        let _ = writeln!(
            svg,
            "\" fill=\"none\" stroke=\"#{red:02x}{green:02x}{blue:02x}\" stroke-opacity=\"{:.17e}\" stroke-width=\"{:.17e}\" stroke-linecap=\"round\" stroke-linejoin=\"round\"/>",
            stroke.color[3], stroke.width
        );
    }
    svg.push_str("</svg>\n");
    Ok(svg.into_bytes())
}

fn parse_projection(view: &str) -> Result<Projection> {
    match view {
        "" | "front" => Ok(Projection::Front),
        "back" => Ok(Projection::Back),
        "right" => Ok(Projection::Right),
        "left" => Ok(Projection::Left),
        "top" => Ok(Projection::Top),
        "bottom" => Ok(Projection::Bottom),
        "iso" | "isometric" => Ok(Projection::Isometric),
        _ => Err(PotError::invalid_argument(format!(
            "unsupported SVG view `{view}`; expected front, back, right, left, top, bottom, iso, or isometric"
        ))),
    }
}

fn write_numbers(output: &mut String, values: [f64; 4]) {
    let _ = write!(
        output,
        "{:.17e} {:.17e} {:.17e} {:.17e}",
        values[0], values[1], values[2], values[3],
    );
}

#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "finite color channels are clamped before byte quantization"
)]
fn color_channels(color: [f64; 4]) -> [u8; 3] {
    [
        (color[0].clamp(0.0, 1.0) * 255.0).round() as u8,
        (color[1].clamp(0.0, 1.0) * 255.0).round() as u8,
        (color[2].clamp(0.0, 1.0) * 255.0).round() as u8,
    ]
}

#[derive(Clone)]
struct ImportedStroke {
    points: Vec<[f64; 2]>,
    cyclic: bool,
    color: [f64; 4],
    fill: Option<[f64; 4]>,
    radius: f64,
}

/// Import SVG `path` and `polyline` geometry into one Grease Pencil object.
pub fn import(path: &Path, scene_id: String) -> Result<ImportedGraph> {
    let bytes = std::fs::read(path).map_err(|error| PotError::io(&error))?;
    let text = String::from_utf8(bytes).map_err(|_| import_error("SVG file is not valid UTF-8"))?;
    let strokes = parse_svg(&text)?;
    let file_name = path
        .file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .filter(|name| !name.is_empty())
        .unwrap_or_else(|| "SVG".to_owned());
    let mut doc = SceneDoc::new(scene_id);
    let node_id = Id::new("svg_node").map_err(|_| import_error("could not create SVG node ID"))?;
    let data_id = Id::new("svg_data").map_err(|_| import_error("could not create SVG data ID"))?;
    let layer_id =
        Id::new("svg_layer").map_err(|_| import_error("could not create SVG layer ID"))?;
    let mut material_ids = Vec::with_capacity(strokes.len());
    let mut imported_strokes = Vec::with_capacity(strokes.len());
    for (index, stroke) in strokes.iter().enumerate() {
        let stroke_id = Id::new(format!("svg_stroke_{index}"))
            .map_err(|_| import_error("SVG contains too many strokes"))?;
        let material_id = Id::new(format!("svg_material_{index}"))
            .map_err(|_| import_error("SVG contains too many strokes"))?;
        doc.materials.insert(
            material_id.clone(),
            Material {
                name: format!("SVG Stroke {}", index + 1),
                base_color: stroke.color,
                metallic: 0.0,
                roughness: 0.8,
                emission_color: [0.0; 3],
                emission_strength: 0.0,
                transmission: 0.0,
                ior: 1.45,
                double_sided: false,
                ..Material::default()
            },
        );
        material_ids.push(material_id.clone());
        let points = stroke
            .points
            .iter()
            .enumerate()
            .map(|(point_index, point)| {
                let time = u32::try_from(point_index)
                    .map_err(|_| import_error("SVG stroke contains too many points"))?;
                Ok(json!({
                    "position": [point[0], point[1], 0.0],
                    "pressure": 1.0,
                    "radius": stroke.radius,
                    "opacity": 1.0,
                    "time": f64::from(time),
                }))
            })
            .collect::<Result<Vec<_>>>()?;
        imported_strokes.push(json!({
            "id": stroke_id.as_str(),
            "material": material_id.as_str(),
            "cyclic": stroke.cyclic,
            "fill": stroke.fill,
            "points": points,
        }));
    }
    let grease_pencil_block: DataBlock = serde_json::from_value(json!({
        "type": "grease_pencil",
        "grease_pencil": {
            "layers": [{
                "id": layer_id.as_str(),
                "name": "SVG",
                "opacity": 1.0,
                "visible": true,
                "frames": [{ "frame": 1.0, "strokes": imported_strokes }]
            }]
        }
    }))
    .map_err(|error| import_error(format!("could not build Grease Pencil data: {error}")))?;
    doc.data_blocks.insert(data_id.clone(), grease_pencil_block);
    doc.nodes.insert(
        node_id.clone(),
        Node {
            name: file_name,
            kind: "grease_pencil".to_owned(),
            transform: Transform::default(),
            data: Some(data_id),
            materials: material_ids,
            ..Node::default()
        },
    );
    let root_id =
        Id::new("collection_root").map_err(|_| import_error("scene root collection is missing"))?;
    let root = doc
        .collections
        .get_mut(&root_id)
        .ok_or_else(|| import_error("scene root collection is missing"))?;
    root.objects.push(node_id);
    doc.validate().map_err(|error| {
        PotError::with_details(
            ErrorCode::ImportFailed,
            "imported SVG scene is invalid",
            json!({ "reason": error.to_string() }),
        )
    })?;
    Ok(ImportedGraph {
        doc,
        losses: Vec::new(),
        id_mappings: json!({}),
        compat_blobs: Vec::new(),
        assets: Vec::new(),
        source: json!({
            "format": "svg",
            "path": path.display().to_string(),
            "file_name": path.file_name().map(|name| name.to_string_lossy().into_owned()),
        }),
    })
}

#[derive(Clone)]
struct ElementContext {
    name: String,
    in_defs: bool,
    opacity: f64,
}

struct ShapeStyle {
    color: Option<[f64; 4]>,
    fill: Option<[f64; 4]>,
    radius: Option<f64>,
    opacity: f64,
    stroke_opacity: f64,
    fill_opacity: f64,
}

impl Default for ShapeStyle {
    fn default() -> Self {
        Self {
            color: None,
            fill: None,
            radius: Some(0.5),
            opacity: 1.0,
            stroke_opacity: 1.0,
            fill_opacity: 1.0,
        }
    }
}

fn parse_svg(input: &str) -> Result<Vec<ImportedStroke>> {
    let input = input.strip_prefix('\u{feff}').unwrap_or(input);
    let mut output = Vec::new();
    let mut stack = Vec::<ElementContext>::new();
    let mut cursor = 0;
    let mut root_seen = false;
    let mut root_closed = false;
    while cursor < input.len() {
        let Some(relative_open) = input[cursor..].find('<') else {
            let text = &input[cursor..];
            validate_xml_text(text, stack.last().map(|context| context.name.as_str()))?;
            break;
        };
        let text = &input[cursor..cursor + relative_open];
        validate_xml_text(text, stack.last().map(|context| context.name.as_str()))?;
        cursor += relative_open;
        if input[cursor..].starts_with("<!--") {
            let Some(end) = input[cursor + 4..].find("-->") else {
                return Err(import_error("unterminated SVG XML comment"));
            };
            if input[cursor + 4..cursor + 4 + end].contains("--") {
                return Err(import_error("malformed SVG XML comment"));
            }
            cursor += 4 + end + 3;
            continue;
        }
        if input[cursor..].starts_with("<?") {
            let Some(end) = input[cursor + 2..].find("?>") else {
                return Err(import_error("unterminated SVG processing instruction"));
            };
            cursor += 2 + end + 2;
            continue;
        }
        if input[cursor..].starts_with("<!") {
            return Err(import_error(
                "SVG declarations and CDATA sections are unsupported",
            ));
        }
        let tag_end = find_tag_end(input, cursor + 1)?;
        let raw_tag = &input[cursor + 1..tag_end];
        cursor = tag_end + 1;
        if let Some(close_name) = raw_tag.strip_prefix('/') {
            let close_name = close_name.trim();
            if !valid_xml_name(close_name) || close_name.contains(char::is_whitespace) {
                return Err(import_error("malformed SVG closing tag"));
            }
            let context = stack
                .pop()
                .ok_or_else(|| import_error("SVG closing tag has no open element"))?;
            if context.name != close_name {
                return Err(import_error("SVG XML elements are not properly nested"));
            }
            if stack.is_empty() {
                root_closed = true;
            }
            continue;
        }
        let (name, attributes, self_closing) = parse_open_tag(raw_tag)?;
        if stack.is_empty() {
            if root_seen || root_closed || name != "svg" {
                return Err(import_error("SVG document must have one svg root element"));
            }
            root_seen = true;
        } else if root_closed {
            return Err(import_error("SVG element appears after the root element"));
        }
        if attributes.contains_key("transform") {
            return Err(import_error("SVG transforms are unsupported"));
        }
        if attributes.contains_key("class") {
            return Err(import_error("SVG CSS classes are unsupported"));
        }
        let parent_in_defs = stack.last().is_some_and(|context| context.in_defs);
        let in_defs = parent_in_defs || name == "defs";
        let parent_opacity = stack.last().map_or(1.0, |context| context.opacity);
        let local_opacity = attributes
            .get("opacity")
            .map(|value| parse_opacity(value))
            .transpose()?
            .unwrap_or(1.0);
        let inherited_opacity = parent_opacity * local_opacity;
        if !inherited_opacity.is_finite() {
            return Err(import_error("SVG opacity is not finite"));
        }

        match name.as_str() {
            "svg" | "g" | "defs" | "title" | "desc" | "metadata" => {
                if attributes.contains_key("style") {
                    return Err(import_error("SVG container CSS styles are unsupported"));
                }
                for key in [
                    "stroke",
                    "stroke-width",
                    "stroke-opacity",
                    "fill",
                    "fill-opacity",
                ] {
                    if name != "svg" && attributes.contains_key(key) {
                        return Err(import_error(
                            "inherited SVG paint styles on containers are unsupported",
                        ));
                    }
                }
                if name == "svg" && stack.iter().any(|context| context.name == "svg") {
                    return Err(import_error("nested SVG roots are unsupported"));
                }
            }
            "path" => {
                let style = parse_shape_style(&attributes, inherited_opacity)?;
                let data = attributes
                    .get("d")
                    .ok_or_else(|| import_error("SVG path is missing its d attribute"))?;
                let geometry = parse_path(data)?;
                if !in_defs {
                    output.extend(geometry.into_iter().map(|(points, cyclic)| ImportedStroke {
                        points,
                        cyclic,
                        color: style.color.unwrap_or([0.0, 0.0, 0.0, 1.0]),
                        fill: style.fill.map(|mut fill| {
                            fill[3] *= style.fill_opacity * inherited_opacity;
                            fill
                        }),
                        radius: style.radius.unwrap_or(0.5),
                    }));
                }
            }
            "polyline" => {
                let style = parse_shape_style(&attributes, inherited_opacity)?;
                let points_data = attributes
                    .get("points")
                    .ok_or_else(|| import_error("SVG polyline is missing its points attribute"))?;
                let values = parse_number_list(points_data)?;
                if values.is_empty() || values.len() % 2 != 0 {
                    return Err(import_error("SVG polyline points must be coordinate pairs"));
                }
                let points = values
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|pair| [pair[0], pair[1]])
                    .collect::<Vec<_>>();
                if !in_defs {
                    output.push(ImportedStroke {
                        points,
                        cyclic: false,
                        color: style.color.unwrap_or([0.0, 0.0, 0.0, 1.0]),
                        fill: style.fill.map(|mut fill| {
                            fill[3] *= style.fill_opacity * inherited_opacity;
                            fill
                        }),
                        radius: style.radius.unwrap_or(0.5),
                    });
                }
            }
            "style" => return Err(import_error("SVG stylesheet elements are unsupported")),
            "polygon" | "line" | "rect" | "circle" | "ellipse" | "text" | "image" | "use"
            | "symbol" | "foreignObject" | "switch" => {
                return Err(import_error(format!("unsupported SVG element <{name}>")));
            }
            _ => return Err(import_error(format!("unsupported SVG element <{name}>"))),
        }
        if !self_closing {
            stack.push(ElementContext {
                name,
                in_defs,
                opacity: inherited_opacity,
            });
        } else if stack.is_empty() {
            root_closed = true;
        }
    }
    if !stack.is_empty() {
        return Err(import_error("SVG document has unclosed elements"));
    }
    if !root_seen || !root_closed {
        return Err(import_error(
            "SVG document has no complete svg root element",
        ));
    }
    Ok(output)
}

fn validate_xml_text(text: &str, parent: Option<&str>) -> Result<()> {
    let decoded = decode_entities(text)?;
    if parent.is_some_and(|name| matches!(name, "title" | "desc" | "metadata"))
        || decoded.trim().is_empty()
    {
        Ok(())
    } else {
        Err(import_error("unexpected text in SVG markup"))
    }
}

fn find_tag_end(input: &str, mut cursor: usize) -> Result<usize> {
    let mut quote = None;
    while cursor < input.len() {
        let character = input.as_bytes()[cursor];
        if let Some(expected) = quote {
            if character == expected {
                quote = None;
            }
        } else if character == b'\'' || character == b'"' {
            quote = Some(character);
        } else if character == b'>' {
            return Ok(cursor);
        } else if character == b'<' {
            return Err(import_error("unexpected < inside an SVG tag"));
        }
        cursor += 1;
    }
    Err(import_error("unterminated SVG tag"))
}

fn parse_open_tag(raw: &str) -> Result<(String, BTreeMap<String, String>, bool)> {
    let bytes = raw.as_bytes();
    let mut cursor = 0;
    skip_ascii_space(bytes, &mut cursor);
    let name_start = cursor;
    while cursor < bytes.len() && !bytes[cursor].is_ascii_whitespace() && bytes[cursor] != b'/' {
        cursor += 1;
    }
    let name = &raw[name_start..cursor];
    if !valid_xml_name(name) {
        return Err(import_error("malformed SVG element name"));
    }
    let mut attributes = BTreeMap::new();
    let mut self_closing = false;
    loop {
        skip_ascii_space(bytes, &mut cursor);
        if cursor == bytes.len() {
            break;
        }
        if bytes[cursor] == b'/' {
            cursor += 1;
            skip_ascii_space(bytes, &mut cursor);
            if cursor != bytes.len() {
                return Err(import_error("malformed self-closing SVG element"));
            }
            self_closing = true;
            break;
        }
        let attribute_start = cursor;
        while cursor < bytes.len()
            && !bytes[cursor].is_ascii_whitespace()
            && bytes[cursor] != b'='
            && bytes[cursor] != b'/'
        {
            cursor += 1;
        }
        let attribute_name = &raw[attribute_start..cursor];
        if !valid_xml_name(attribute_name) {
            return Err(import_error("malformed SVG attribute name"));
        }
        skip_ascii_space(bytes, &mut cursor);
        if bytes.get(cursor) != Some(&b'=') {
            return Err(import_error("SVG attributes must have quoted values"));
        }
        cursor += 1;
        skip_ascii_space(bytes, &mut cursor);
        let quote = *bytes
            .get(cursor)
            .ok_or_else(|| import_error("SVG attribute is missing a value"))?;
        if quote != b'\'' && quote != b'"' {
            return Err(import_error("SVG attribute values must be quoted"));
        }
        cursor += 1;
        let value_start = cursor;
        while cursor < bytes.len() && bytes[cursor] != quote {
            if bytes[cursor] == b'<' {
                return Err(import_error("unexpected < inside an SVG attribute"));
            }
            cursor += 1;
        }
        if cursor == bytes.len() {
            return Err(import_error("unterminated SVG attribute value"));
        }
        let value = decode_entities(&raw[value_start..cursor])?;
        cursor += 1;
        if attributes
            .insert(attribute_name.to_owned(), value)
            .is_some()
        {
            return Err(import_error("duplicate SVG attribute"));
        }
    }
    Ok((name.to_owned(), attributes, self_closing))
}

fn valid_xml_name(name: &str) -> bool {
    let mut bytes = name.bytes();
    let Some(first) = bytes.next() else {
        return false;
    };
    (first.is_ascii_alphabetic() || matches!(first, b'_' | b':'))
        && bytes
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b':' | b'-' | b'.'))
}

fn skip_ascii_space(bytes: &[u8], cursor: &mut usize) {
    while bytes.get(*cursor).is_some_and(u8::is_ascii_whitespace) {
        *cursor += 1;
    }
}

fn decode_entities(value: &str) -> Result<String> {
    let mut decoded = String::with_capacity(value.len());
    let mut cursor = 0;
    while let Some(relative_ampersand) = value[cursor..].find('&') {
        let ampersand = cursor + relative_ampersand;
        decoded.push_str(&value[cursor..ampersand]);
        let Some(relative_semicolon) = value[ampersand + 1..].find(';') else {
            return Err(import_error("unterminated XML entity"));
        };
        let semicolon = ampersand + 1 + relative_semicolon;
        let entity = &value[ampersand + 1..semicolon];
        let character = match entity {
            "amp" => '&',
            "lt" => '<',
            "gt" => '>',
            "quot" => '"',
            "apos" => '\'',
            _ if entity.starts_with("#x") || entity.starts_with("#X") => {
                let code = u32::from_str_radix(&entity[2..], 16)
                    .map_err(|_| import_error("invalid hexadecimal XML character reference"))?;
                char::from_u32(code)
                    .ok_or_else(|| import_error("invalid XML character reference"))?
            }
            _ if entity.starts_with('#') => {
                let code = entity[1..]
                    .parse::<u32>()
                    .map_err(|_| import_error("invalid decimal XML character reference"))?;
                char::from_u32(code)
                    .ok_or_else(|| import_error("invalid XML character reference"))?
            }
            _ => return Err(import_error("unsupported XML entity")),
        };
        decoded.push(character);
        cursor = semicolon + 1;
    }
    decoded.push_str(&value[cursor..]);
    Ok(decoded)
}

fn parse_shape_style(
    attributes: &BTreeMap<String, String>,
    inherited_opacity: f64,
) -> Result<ShapeStyle> {
    let mut style = ShapeStyle::default();
    let mut declarations = BTreeMap::new();
    if let Some(raw_style) = attributes.get("style") {
        for declaration in raw_style.split(';').filter(|part| !part.trim().is_empty()) {
            let (name, value) = declaration
                .split_once(':')
                .ok_or_else(|| import_error("malformed SVG style declaration"))?;
            let name = name.trim();
            let value = value.trim();
            if declarations
                .insert(name.to_owned(), value.to_owned())
                .is_some()
            {
                return Err(import_error("duplicate SVG style declaration"));
            }
        }
    }
    for (name, value) in &declarations {
        if !matches!(
            name.as_str(),
            "stroke"
                | "stroke-width"
                | "stroke-opacity"
                | "fill"
                | "fill-opacity"
                | "opacity"
                | "stroke-linecap"
                | "stroke-linejoin"
        ) {
            return Err(import_error(format!(
                "unsupported SVG style property `{name}`"
            )));
        }
        if (name == "stroke-linecap" || name == "stroke-linejoin")
            && !matches!(
                value.as_str(),
                "round" | "butt" | "square" | "miter" | "bevel"
            )
        {
            return Err(import_error("unsupported SVG stroke cap or join"));
        }
    }
    let style_value = |key: &str| -> Option<&str> {
        declarations
            .get(key)
            .map(String::as_str)
            .or_else(|| attributes.get(key).map(String::as_str))
    };
    if let Some(value) = style_value("stroke") {
        if value == "none" {
            return Err(import_error("SVG strokes with stroke=none are unsupported"));
        }
        style.color = Some(parse_color(value)?);
    }
    if let Some(value) = style_value("fill")
        && value != "none"
    {
        style.fill = Some(parse_color(value)?);
    }
    if let Some(value) = style_value("stroke-width") {
        let value = parse_svg_length(value)?;
        if value < 0.0 {
            return Err(import_error("SVG stroke-width cannot be negative"));
        }
        style.radius = Some(value * 0.5);
    }
    if let Some(value) = style_value("opacity") {
        style.opacity = parse_opacity(value)?;
    }
    if let Some(value) = style_value("stroke-opacity") {
        style.stroke_opacity = parse_opacity(value)?;
    }
    if let Some(value) = style_value("fill-opacity") {
        style.fill_opacity = parse_opacity(value)?;
    }
    let mut color = style.color.unwrap_or([0.0, 0.0, 0.0, 1.0]);
    color[3] *= style.opacity * style.stroke_opacity * inherited_opacity;
    style.color = Some(color);
    Ok(style)
}

fn parse_svg_length(value: &str) -> Result<f64> {
    let value = value.trim().strip_suffix("px").unwrap_or(value.trim());
    let parsed = value
        .parse::<f64>()
        .map_err(|_| import_error("unsupported SVG length syntax"))?;
    if !parsed.is_finite() {
        return Err(import_error("SVG length is not finite"));
    }
    Ok(parsed)
}

fn parse_opacity(value: &str) -> Result<f64> {
    let parsed = value
        .trim()
        .parse::<f64>()
        .map_err(|_| import_error("invalid SVG opacity"))?;
    if !parsed.is_finite() {
        return Err(import_error("SVG opacity is not finite"));
    }
    Ok(parsed.clamp(0.0, 1.0))
}

fn parse_color(value: &str) -> Result<[f64; 4]> {
    let value = value.trim();
    if let Some(hex) = value.strip_prefix('#') {
        if !hex.is_ascii() {
            return Err(import_error("invalid SVG hexadecimal color"));
        }
        let (red, green, blue, alpha) = match hex.len() {
            3 => (
                parse_hex_pair(&hex[0..1], true)?,
                parse_hex_pair(&hex[1..2], true)?,
                parse_hex_pair(&hex[2..3], true)?,
                255,
            ),
            4 => (
                parse_hex_pair(&hex[0..1], true)?,
                parse_hex_pair(&hex[1..2], true)?,
                parse_hex_pair(&hex[2..3], true)?,
                parse_hex_pair(&hex[3..4], true)?,
            ),
            6 => (
                parse_hex_pair(&hex[0..2], false)?,
                parse_hex_pair(&hex[2..4], false)?,
                parse_hex_pair(&hex[4..6], false)?,
                255,
            ),
            8 => (
                parse_hex_pair(&hex[0..2], false)?,
                parse_hex_pair(&hex[2..4], false)?,
                parse_hex_pair(&hex[4..6], false)?,
                parse_hex_pair(&hex[6..8], false)?,
            ),
            _ => return Err(import_error("unsupported SVG hexadecimal color")),
        };
        return Ok([
            f64::from(red) / 255.0,
            f64::from(green) / 255.0,
            f64::from(blue) / 255.0,
            f64::from(alpha) / 255.0,
        ]);
    }
    let (red, green, blue, alpha) = match value.to_ascii_lowercase().as_str() {
        "black" => (0, 0, 0, 255),
        "white" => (255, 255, 255, 255),
        "red" => (255, 0, 0, 255),
        "green" => (0, 128, 0, 255),
        "blue" => (0, 0, 255, 255),
        "yellow" => (255, 255, 0, 255),
        "cyan" | "aqua" => (0, 255, 255, 255),
        "magenta" | "fuchsia" => (255, 0, 255, 255),
        "gray" | "grey" => (128, 128, 128, 255),
        "transparent" => (0, 0, 0, 0),
        _ => return Err(import_error("unsupported SVG color syntax")),
    };
    Ok([
        f64::from(red) / 255.0,
        f64::from(green) / 255.0,
        f64::from(blue) / 255.0,
        f64::from(alpha) / 255.0,
    ])
}
fn parse_hex_pair(value: &str, shorthand: bool) -> Result<u8> {
    let parsed =
        u8::from_str_radix(value, 16).map_err(|_| import_error("invalid SVG hexadecimal color"))?;
    Ok(if shorthand { parsed * 17 } else { parsed })
}

fn parse_number_list(input: &str) -> Result<Vec<f64>> {
    let bytes = input.as_bytes();
    let mut cursor = 0;
    let mut values = Vec::new();
    let mut previous_was_number = false;
    loop {
        skip_ascii_space(bytes, &mut cursor);
        if cursor >= bytes.len() {
            break;
        }
        if bytes[cursor] == b',' {
            if !previous_was_number {
                return Err(import_error("malformed SVG number separator"));
            }
            cursor += 1;
            skip_ascii_space(bytes, &mut cursor);
            if cursor >= bytes.len() || !is_number_start(bytes[cursor]) {
                return Err(import_error("malformed SVG number separator"));
            }
        }
        let value = parse_number(bytes, &mut cursor)?;
        values.push(value);
        previous_was_number = true;
        if cursor < bytes.len()
            && !bytes[cursor].is_ascii_whitespace()
            && bytes[cursor] != b','
            && !matches!(bytes[cursor], b'+' | b'-' | b'.')
        {
            return Err(import_error("invalid SVG number list"));
        }
    }
    Ok(values)
}

#[derive(Clone, Copy)]
enum PathToken {
    Command(u8),
    Number(f64),
}

fn tokenize_path(input: &str) -> Result<Vec<PathToken>> {
    let bytes = input.as_bytes();
    let mut cursor = 0;
    let mut tokens = Vec::new();
    let mut previous_was_number = false;
    loop {
        skip_ascii_space(bytes, &mut cursor);
        if cursor >= bytes.len() {
            break;
        }
        if bytes[cursor] == b',' {
            if !previous_was_number {
                return Err(import_error("malformed SVG path separator"));
            }
            cursor += 1;
            skip_ascii_space(bytes, &mut cursor);
            if cursor >= bytes.len() || !is_number_start(bytes[cursor]) {
                return Err(import_error("malformed SVG path separator"));
            }
        }
        if bytes[cursor].is_ascii_alphabetic() {
            tokens.push(PathToken::Command(bytes[cursor]));
            cursor += 1;
            previous_was_number = false;
        } else if is_number_start(bytes[cursor]) {
            tokens.push(PathToken::Number(parse_number(bytes, &mut cursor)?));
            previous_was_number = true;
        } else {
            return Err(import_error("invalid SVG path data"));
        }
        if cursor < bytes.len()
            && !bytes[cursor].is_ascii_whitespace()
            && bytes[cursor] != b','
            && !bytes[cursor].is_ascii_alphabetic()
            && !matches!(bytes[cursor], b'+' | b'-' | b'.')
        {
            return Err(import_error("invalid SVG path data"));
        }
    }
    Ok(tokens)
}

fn is_number_start(byte: u8) -> bool {
    byte.is_ascii_digit() || matches!(byte, b'+' | b'-' | b'.')
}

fn parse_number(bytes: &[u8], cursor: &mut usize) -> Result<f64> {
    let start = *cursor;
    if bytes
        .get(*cursor)
        .is_some_and(|byte| matches!(byte, b'+' | b'-'))
    {
        *cursor += 1;
    }
    let mut digits = 0;
    while bytes.get(*cursor).is_some_and(u8::is_ascii_digit) {
        *cursor += 1;
        digits += 1;
    }
    if bytes.get(*cursor) == Some(&b'.') {
        *cursor += 1;
        while bytes.get(*cursor).is_some_and(u8::is_ascii_digit) {
            *cursor += 1;
            digits += 1;
        }
    }
    if digits == 0 {
        return Err(import_error("invalid SVG number"));
    }
    if bytes
        .get(*cursor)
        .is_some_and(|byte| matches!(byte, b'e' | b'E'))
    {
        *cursor += 1;
        if bytes
            .get(*cursor)
            .is_some_and(|byte| matches!(byte, b'+' | b'-'))
        {
            *cursor += 1;
        }
        let exponent_start = *cursor;
        while bytes.get(*cursor).is_some_and(u8::is_ascii_digit) {
            *cursor += 1;
        }
        if *cursor == exponent_start {
            return Err(import_error("invalid SVG number exponent"));
        }
    }
    let text = std::str::from_utf8(&bytes[start..*cursor])
        .map_err(|_| import_error("invalid SVG number encoding"))?;
    let value = text
        .parse::<f64>()
        .map_err(|_| import_error("invalid SVG number"))?;
    if !value.is_finite() {
        return Err(import_error("SVG geometry contains non-finite coordinates"));
    }
    Ok(value)
}

fn parse_path(input: &str) -> Result<Vec<(Vec<[f64; 2]>, bool)>> {
    let tokens = tokenize_path(input)?;
    if tokens.is_empty() {
        return Err(import_error("SVG path data is empty"));
    }
    let mut paths = Vec::<(Vec<[f64; 2]>, bool)>::new();
    let mut cursor = 0;
    let mut command = None;
    let mut current = [0.0, 0.0];
    let mut subpath_start = [0.0, 0.0];
    while cursor < tokens.len() {
        if let PathToken::Command(next_command) = tokens[cursor] {
            command = Some(next_command);
            cursor += 1;
            if matches!(next_command, b'Z' | b'z') {
                let Some((points, cyclic)) = paths.last_mut() else {
                    return Err(import_error("SVG closepath has no active subpath"));
                };
                if points.is_empty() {
                    return Err(import_error("SVG closepath has no active subpath"));
                }
                *cyclic = true;
                current = subpath_start;
                command = None;
                continue;
            }
        }
        let active_command =
            command.ok_or_else(|| import_error("SVG path numbers are missing a command"))?;
        let start = cursor;
        while cursor < tokens.len() && matches!(tokens[cursor], PathToken::Number(_)) {
            cursor += 1;
        }
        if start == cursor {
            return Err(import_error("SVG path command is missing parameters"));
        }
        let values = tokens[start..cursor]
            .iter()
            .map(|token| match token {
                PathToken::Number(value) => Ok(*value),
                PathToken::Command(_) => Err(import_error("invalid SVG path parameters")),
            })
            .collect::<Result<Vec<_>>>()?;
        let relative = active_command.is_ascii_lowercase();
        match active_command.to_ascii_uppercase() {
            b'M' => {
                if values.len() < 2 || values.len() % 2 != 0 {
                    return Err(import_error("SVG moveto requires coordinate pairs"));
                }
                let first = path_point([values[0], values[1]], current, relative)?;
                current = first;
                subpath_start = first;
                paths.push((vec![first], false));
                for pair in values[2..].as_chunks::<2>().0 {
                    let point = path_point([pair[0], pair[1]], current, relative)?;
                    if let Some((points, _)) = paths.last_mut() {
                        points.push(point);
                    }
                    current = point;
                }
                command = Some(if relative { b'l' } else { b'L' });
            }
            b'L' => {
                if values.len() < 2 || values.len() % 2 != 0 {
                    return Err(import_error("SVG lineto requires coordinate pairs"));
                }
                if paths.is_empty() {
                    return Err(import_error("SVG lineto requires an active subpath"));
                }
                for pair in values.as_chunks::<2>().0 {
                    let point = path_point([pair[0], pair[1]], current, relative)?;
                    if let Some((points, _)) = paths.last_mut() {
                        points.push(point);
                    }
                    current = point;
                }
            }
            b'C' => {
                if values.len() < 6 || values.len() % 6 != 0 {
                    return Err(import_error("SVG cubic command requires six-value groups"));
                }
                if paths.is_empty() {
                    return Err(import_error("SVG cubic command requires an active subpath"));
                }
                for group in values.as_chunks::<6>().0 {
                    let origin = current;
                    let control1 = path_point([group[0], group[1]], origin, relative)?;
                    let control2 = path_point([group[2], group[3]], origin, relative)?;
                    let end = path_point([group[4], group[5]], origin, relative)?;
                    for step in 1..=CUBIC_STEPS {
                        let t = f64::from(step) / f64::from(CUBIC_STEPS);
                        let u = 1.0 - t;
                        let point = [
                            u * u * u * origin[0]
                                + 3.0 * u * u * t * control1[0]
                                + 3.0 * u * t * t * control2[0]
                                + t * t * t * end[0],
                            u * u * u * origin[1]
                                + 3.0 * u * u * t * control1[1]
                                + 3.0 * u * t * t * control2[1]
                                + t * t * t * end[1],
                        ];
                        if point.iter().any(|coordinate| !coordinate.is_finite()) {
                            return Err(import_error("SVG cubic curve is not finite"));
                        }
                        if let Some((points, _)) = paths.last_mut() {
                            points.push(point);
                        }
                    }
                    current = end;
                }
            }
            _ => {
                return Err(import_error(format!(
                    "unsupported SVG path command `{}`",
                    char::from(active_command)
                )));
            }
        }
    }
    Ok(paths)
}

fn path_point(value: [f64; 2], current: [f64; 2], relative: bool) -> Result<[f64; 2]> {
    let point = if relative {
        [current[0] + value[0], current[1] + value[1]]
    } else {
        value
    };
    if point.iter().any(|coordinate| !coordinate.is_finite()) {
        Err(import_error("SVG path coordinate is not finite"))
    } else {
        Ok(point)
    }
}
