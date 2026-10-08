use std::{
    fs,
    io::{Read, Write},
    path::Path,
};

use flate2::{Compression, read::ZlibDecoder, write::ZlibEncoder};
use serde_json::json;

use crate::{
    error::{ErrorCode, PotError, Result},
    eval::Snapshot,
    model::SceneDoc,
};

use super::{ImportedGraph, import_error};

const MAGIC: &[u8; 23] = b"Kaydara FBX Binary  \0\x1a\0";
const FBX_VERSION: u32 = 7400;
const MAX_FILE_BYTES: usize = 1_073_741_824;

const READ_ERRORS: super::LeReadErrors = super::LeReadErrors {
    offset_overflow: "FBX offset overflow",
    out_of_bounds: "FBX file is truncated",
    invalid_integer: "FBX integer is truncated",
};

#[derive(Clone, Debug)]
struct Arg {
    value: String,
    is_string: bool,
}

#[derive(Clone, Debug)]
struct Node {
    name: String,
    args: Vec<Arg>,
    children: Vec<Node>,
}

/// Export an FBX 7.4 binary file from the ASCII adapter's complete supported graph.
pub(crate) fn export(doc: &SceneDoc, snapshot: &Snapshot) -> Result<Vec<u8>> {
    let ascii = super::fbx::export(doc, snapshot)?;
    let text = std::str::from_utf8(&ascii)
        .map_err(|error| PotError::new(ErrorCode::InternalError, error.to_string()))?;
    let roots = parse_ascii(text)?;
    encode_binary(&roots, FBX_VERSION)
}

/// Import either FBX 7.4 or 7.5 binary data by decoding its node/property tree into the
/// ASCII representation consumed by the shared FBX graph importer.
pub(crate) fn import(file: &Path, scene_id: String) -> Result<ImportedGraph> {
    let bytes = fs::read(file).map_err(|error| PotError::io(&error))?;
    if bytes.len() > MAX_FILE_BYTES {
        return Err(PotError::new(
            ErrorCode::LimitExceeded,
            "FBX file exceeds the size limit",
        ));
    }
    let ascii = if bytes.starts_with(MAGIC) {
        let (roots, version) = decode_binary(&bytes)?;
        if !(7400..=7500).contains(&version) {
            return Err(PotError::with_details(
                ErrorCode::UnsupportedVersion,
                "FBX binary file version is outside the supported 7.4/7.5 range",
                json!({"version": version}),
            ));
        }
        render_ascii(&roots).into_bytes()
    } else {
        bytes
    };
    super::fbx::import_bytes(&ascii, scene_id)
}

fn parse_ascii(text: &str) -> Result<Vec<Node>> {
    let tokens = tokenize_ascii(text)?;
    let mut cursor = 0;
    let nodes = parse_node_list(&tokens, &mut cursor, false)?;
    if cursor != tokens.len() {
        return Err(export_error("ASCII FBX has trailing unmatched delimiters"));
    }
    Ok(nodes)
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum Token {
    Atom(String),
    String(String),
    Colon,
    Comma,
    Open,
    Close,
    Star,
    Newline,
}

fn tokenize_ascii(text: &str) -> Result<Vec<Token>> {
    let mut tokens = Vec::new();
    let mut characters = text.chars().peekable();
    while let Some(character) = characters.next() {
        match character {
            '\n' | '\r' => {
                if character == '\r' && characters.peek() == Some(&'\n') {
                    characters.next();
                }
                tokens.push(Token::Newline);
            }
            ' ' | '\t' => {}
            ';' => {
                for next in characters.by_ref() {
                    if next == '\n' {
                        tokens.push(Token::Newline);
                        break;
                    }
                }
            }
            ':' => tokens.push(Token::Colon),
            ',' => tokens.push(Token::Comma),
            '{' => tokens.push(Token::Open),
            '}' => tokens.push(Token::Close),
            '*' => tokens.push(Token::Star),
            '"' => {
                let mut value = String::new();
                let mut terminated = false;
                while let Some(next) = characters.next() {
                    match next {
                        '"' => {
                            terminated = true;
                            break;
                        }
                        '\\' => match characters.next() {
                            Some('"') => value.push('"'),
                            Some('\\') => value.push('\\'),
                            Some(other) => {
                                value.push('\\');
                                value.push(other);
                            }
                            None => {
                                return Err(export_error("ASCII FBX string has a trailing escape"));
                            }
                        },
                        other => value.push(other),
                    }
                }
                if !terminated {
                    return Err(export_error("ASCII FBX string is unterminated"));
                }
                tokens.push(Token::String(value));
            }
            other => {
                let mut value = String::new();
                value.push(other);
                while let Some(next) = characters.peek().copied() {
                    if next.is_whitespace()
                        || matches!(next, ':' | ',' | '{' | '}' | '*' | ';' | '"')
                    {
                        break;
                    }
                    value.push(next);
                    characters.next();
                }
                tokens.push(Token::Atom(value));
            }
        }
    }
    Ok(tokens)
}

fn parse_node_list(tokens: &[Token], cursor: &mut usize, in_block: bool) -> Result<Vec<Node>> {
    let mut nodes = Vec::new();
    loop {
        while matches!(tokens.get(*cursor), Some(Token::Newline)) {
            *cursor += 1;
        }
        match tokens.get(*cursor) {
            Some(Token::Close) if in_block => {
                *cursor += 1;
                return Ok(nodes);
            }
            None if !in_block => return Ok(nodes),
            None => return Err(export_error("ASCII FBX block is unterminated")),
            Some(Token::Close) => {
                return Err(export_error("ASCII FBX has an unmatched closing brace"));
            }
            _ => {}
        }
        let name = match tokens.get(*cursor) {
            Some(Token::Atom(value) | Token::String(value)) => value.clone(),
            _ => return Err(export_error("ASCII FBX node name is invalid")),
        };
        *cursor += 1;
        if !matches!(tokens.get(*cursor), Some(Token::Colon)) {
            return Err(export_error("ASCII FBX node has no colon"));
        }
        *cursor += 1;
        let mut args = Vec::new();
        let mut block = false;
        loop {
            match tokens.get(*cursor) {
                Some(Token::Newline) => {
                    *cursor += 1;
                    break;
                }
                Some(Token::Open) => {
                    *cursor += 1;
                    block = true;
                    break;
                }
                Some(Token::Close) | None => break,
                Some(Token::Comma | Token::Star) => *cursor += 1,
                Some(Token::Atom(value)) => {
                    args.push(Arg {
                        value: value.clone(),
                        is_string: false,
                    });
                    *cursor += 1;
                }
                Some(Token::String(value)) => {
                    args.push(Arg {
                        value: value.clone(),
                        is_string: true,
                    });
                    *cursor += 1;
                }
                Some(Token::Colon) => {
                    return Err(export_error("ASCII FBX node has an unexpected colon"));
                }
            }
        }
        let children = if block {
            parse_node_list(tokens, cursor, true)?
        } else {
            Vec::new()
        };
        nodes.push(Node {
            name,
            args,
            children,
        });
    }
}

fn encode_binary(roots: &[Node], version: u32) -> Result<Vec<u8>> {
    let wide = version >= 7500;
    let sentinel_len = if wide { 25 } else { 13 };
    let mut bytes = Vec::with_capacity(64 * 1024);
    bytes.extend_from_slice(MAGIC);
    bytes.extend_from_slice(&version.to_le_bytes());
    for node in roots {
        encode_node(node, None, wide, &mut bytes)?;
    }
    bytes.resize(
        bytes
            .len()
            .checked_add(sentinel_len)
            .ok_or_else(|| limit_error("FBX offset overflow"))?,
        0,
    );
    bytes.extend_from_slice(&version.to_le_bytes());
    bytes.resize(
        bytes
            .len()
            .checked_add(120)
            .ok_or_else(|| limit_error("FBX footer overflow"))?,
        0,
    );
    bytes.extend_from_slice(&[
        0xfa, 0xbc, 0xab, 0x09, 0xd0, 0xc8, 0xd4, 0x66, 0xb1, 0x76, 0xfb, 0x83, 0x1c, 0xf7, 0x26,
        0x7e,
    ]);
    bytes.extend_from_slice(&0_u32.to_le_bytes());
    Ok(bytes)
}

fn encode_node(node: &Node, parent: Option<&str>, wide: bool, output: &mut Vec<u8>) -> Result<()> {
    let start = output.len();
    let name = node.name.as_bytes();
    let name_len =
        u8::try_from(name.len()).map_err(|_| limit_error("FBX node name is too long"))?;
    let header_len = if wide { 25_usize } else { 13_usize };
    let header_at = output.len();
    output.resize(header_at + header_len, 0);
    output.extend_from_slice(name);
    let property_start = output.len();
    let has_array_properties = node.children.iter().any(|child| child.name == "a");
    let scalar_args = if has_array_properties {
        &node.args[..0]
    } else {
        node.args.as_slice()
    };
    for (index, arg) in scalar_args.iter().enumerate() {
        let encoded_arg = binary_object_name_arg(node, index, arg);
        encode_scalar(&encoded_arg, property_kind(node, index, parent)?, output)?;
    }
    let array_count = node
        .children
        .iter()
        .filter(|child| child.name == "a")
        .count();
    for child in node.children.iter().filter(|child| child.name == "a") {
        encode_array(&child.args, array_type(&node.name), output)?;
    }
    let property_end = output.len();
    for child in &node.children {
        if child.name != "a" {
            encode_node(child, Some(&node.name), wide, output)?;
        }
    }
    let sentinel = if wide { 25 } else { 13 };
    output.resize(
        output
            .len()
            .checked_add(sentinel)
            .ok_or_else(|| limit_error("FBX child table overflow"))?,
        0,
    );
    let end = output.len();
    let property_count = u32::try_from(
        scalar_args
            .len()
            .checked_add(array_count)
            .ok_or_else(|| limit_error("FBX property count overflow"))?,
    )
    .map_err(|_| limit_error("FBX property count is too large"))?;
    let property_len = u32::try_from(property_end - property_start)
        .map_err(|_| limit_error("FBX property list is too large"))?;
    if wide {
        write_u64_at(
            output,
            header_at,
            u64::try_from(end).map_err(|_| limit_error("FBX offset overflow"))?,
        );
        write_u64_at(output, header_at + 8, u64::from(property_count));
        write_u64_at(output, header_at + 16, u64::from(property_len));
    } else {
        write_u32_at(
            output,
            header_at,
            u32::try_from(end).map_err(|_| limit_error("FBX 7.4 offset overflow"))?,
        );
        write_u32_at(output, header_at + 4, property_count);
        write_u32_at(output, header_at + 8, property_len);
    }
    let encoded_name_len_at = header_at + if wide { 24 } else { 12 };
    output[encoded_name_len_at] = name_len;
    if end <= start {
        return Err(PotError::new(
            ErrorCode::InternalError,
            "FBX node has an invalid end offset",
        ));
    }
    Ok(())
}

fn binary_object_name_arg(node: &Node, index: usize, arg: &Arg) -> Arg {
    if index == 1
        && let Some((class, name)) = arg.value.split_once("::")
    {
        let binary_class = match class {
            "AnimationStack" => "AnimStack",
            "AnimationLayer" => "AnimLayer",
            "AnimationCurveNode" => "AnimCurveNode",
            "AnimationCurve" => "AnimCurve",
            "SubDeformer" => "Deformer",
            class => class,
        };
        if class == node.name
            || binary_class == node.name
            || matches!(
                (class, node.name.as_str()),
                ("AnimStack", "AnimationStack")
                    | ("AnimLayer", "AnimationLayer")
                    | ("AnimCurveNode", "AnimationCurveNode")
                    | ("AnimCurve", "AnimationCurve")
            )
        {
            return Arg {
                value: format!("{name}\0\x01{binary_class}"),
                is_string: true,
            };
        }
    }
    arg.clone()
}

fn encode_scalar(arg: &Arg, kind: char, output: &mut Vec<u8>) -> Result<()> {
    output.push(kind as u8);
    match kind {
        'S' => {
            let bytes = arg.value.as_bytes();
            output.extend_from_slice(
                &u32::try_from(bytes.len())
                    .map_err(|_| limit_error("FBX string is too long"))?
                    .to_le_bytes(),
            );
            output.extend_from_slice(bytes);
        }
        'L' => output.extend_from_slice(&parse_i64(&arg.value)?.to_le_bytes()),
        'I' => {
            let value = i32::try_from(parse_i64(&arg.value)?)
                .map_err(|_| export_error("FBX integer exceeds 32-bit range"))?;
            output.extend_from_slice(&value.to_le_bytes());
        }
        'C' => {
            let value = match arg.value.as_str() {
                "T" => true,
                "F" => false,
                _ => parse_i64(&arg.value)? != 0,
            };
            output.push(u8::from(value));
        }
        'F' => output.extend_from_slice(&(parse_f64(&arg.value)? as f32).to_le_bytes()),
        'D' => output.extend_from_slice(&parse_f64(&arg.value)?.to_le_bytes()),
        _ => return Err(export_error("unsupported FBX binary scalar property")),
    }
    Ok(())
}

fn encode_array(args: &[Arg], element_type: char, output: &mut Vec<u8>) -> Result<()> {
    let mut raw = Vec::new();
    for value in args {
        match element_type {
            'd' => raw.extend_from_slice(&parse_f64(&value.value)?.to_le_bytes()),
            'f' => raw.extend_from_slice(&(parse_f64(&value.value)? as f32).to_le_bytes()),
            'i' => {
                let integer = i32::try_from(parse_i64(&value.value)?)
                    .map_err(|_| export_error("FBX array integer exceeds 32-bit range"))?;
                raw.extend_from_slice(&integer.to_le_bytes());
            }
            'l' => raw.extend_from_slice(&parse_i64(&value.value)?.to_le_bytes()),
            _ => return Err(export_error("unsupported FBX binary array element type")),
        }
    }
    let mut encoder = ZlibEncoder::new(Vec::new(), Compression::default());
    encoder
        .write_all(&raw)
        .map_err(|error| PotError::io(&error))?;
    let compressed = encoder.finish().map_err(|error| PotError::io(&error))?;
    output.push(element_type as u8);
    output.extend_from_slice(
        &u32::try_from(args.len())
            .map_err(|_| limit_error("FBX array is too long"))?
            .to_le_bytes(),
    );
    output.extend_from_slice(&1_u32.to_le_bytes());
    output.extend_from_slice(
        &u32::try_from(compressed.len())
            .map_err(|_| limit_error("compressed FBX array is too large"))?
            .to_le_bytes(),
    );
    output.extend_from_slice(&compressed);
    Ok(())
}

fn property_kind(node: &Node, index: usize, parent: Option<&str>) -> Result<char> {
    if parent == Some("Objects") && index < 3 {
        return Ok(if index == 0 { 'L' } else { 'S' });
    }

    let arg = node.args.get(index).ok_or_else(|| {
        PotError::new(ErrorCode::InternalError, "FBX property argument is missing")
    })?;
    if arg.is_string {
        return Ok('S');
    }
    if node.name == "Shading" {
        return Ok('C');
    }
    if matches!(
        node.name.as_str(),
        "Model"
            | "Geometry"
            | "Deformer"
            | "AnimationStack"
            | "AnimationLayer"
            | "AnimationCurveNode"
            | "AnimationCurve"
            | "Document"
    ) && index == 0
    {
        return Ok('L');
    }
    if node.name == "C" {
        return Ok(if index == 0 {
            'S'
        } else if index == 1 || index == 2 {
            'L'
        } else {
            'S'
        });
    }
    if node.name == "P" {
        return Ok(if index < 4 {
            'S'
        } else {
            match node.args.get(1).map(|arg| arg.value.as_str()) {
                Some("int" | "Integer") => 'I',
                Some("bool" | "Boolean") => 'C',
                _ => 'D',
            }
        });
    }
    if node.name == "ObjectType"
        || node.name == "Property"
        || node.name == "Type"
        || node.name == "TypedIndex"
    {
        return Ok(if index == 0 && arg.value.parse::<i64>().is_err() {
            'S'
        } else {
            'I'
        });
    }
    if matches!(
        node.name.as_str(),
        "FBXHeaderVersion"
            | "FBXVersion"
            | "Version"
            | "Count"
            | "RootNode"
            | "GeometryVersion"
            | "MultiLayer"
            | "Mute"
            | "Solo"
            | "Lock"
            | "BlendMode"
            | "TypedIndex"
    ) {
        return Ok('I');
    }
    if parent == Some("Definitions") && node.name == "Count" {
        return Ok('I');
    }
    if node.name == "a" {
        return Ok(array_type(parent.unwrap_or_default()));
    }
    Ok('D')
}

fn array_type(parent: &str) -> char {
    match parent {
        "PolygonVertexIndex" | "Materials" | "Indexes" | "KeyAttrFlags" | "KeyAttrRefCount" => 'i',
        "KeyTime" => 'l',
        "KeyValueFloat" | "KeyAttrDataFloat" => 'f',
        _ => 'd',
    }
}

fn decode_binary(bytes: &[u8]) -> Result<(Vec<Node>, u32)> {
    if bytes.len() < MAGIC.len() + 4 || !bytes.starts_with(MAGIC) {
        return Err(import_error("FBX binary header is invalid"));
    }
    let version = super::read_u32(bytes, MAGIC.len(), READ_ERRORS)?;
    if !(7000..=8000).contains(&version) {
        return Err(PotError::with_details(
            ErrorCode::UnsupportedVersion,
            "unsupported FBX binary node version",
            json!({"version":version}),
        ));
    }
    let mut cursor = MAGIC.len() + 4;
    let mut roots = Vec::new();
    while cursor < bytes.len() {
        match decode_node(bytes, &mut cursor, version)? {
            Some(node) => roots.push(node),
            None => break,
        }
    }
    Ok((roots, version))
}

fn decode_node(bytes: &[u8], cursor: &mut usize, version: u32) -> Result<Option<Node>> {
    let wide = version >= 7500;
    let header_len = if wide { 25 } else { 13 };
    if bytes.len().saturating_sub(*cursor) < header_len {
        return Err(import_error("FBX node header is truncated"));
    }
    let (end_offset, property_count, property_len, name_length) = if wide {
        let end = super::read_u64(bytes, *cursor, READ_ERRORS)?;
        let count = super::read_u64(bytes, *cursor + 8, READ_ERRORS)?;
        let props = super::read_u64(bytes, *cursor + 16, READ_ERRORS)?;
        let name_len = bytes[*cursor + 24] as usize;
        (
            usize::try_from(end).map_err(|_| import_error("FBX offset is too large"))?,
            usize::try_from(count).map_err(|_| import_error("FBX property count is too large"))?,
            usize::try_from(props).map_err(|_| import_error("FBX property list is too large"))?,
            name_len,
        )
    } else {
        (
            super::read_u32(bytes, *cursor, READ_ERRORS)? as usize,
            super::read_u32(bytes, *cursor + 4, READ_ERRORS)? as usize,
            super::read_u32(bytes, *cursor + 8, READ_ERRORS)? as usize,
            bytes[*cursor + 12] as usize,
        )
    };
    if end_offset == 0 && property_count == 0 && property_len == 0 && name_length == 0 {
        *cursor += header_len;
        return Ok(None);
    }
    let start = *cursor;
    *cursor = cursor
        .checked_add(header_len)
        .ok_or_else(|| import_error("FBX cursor overflow"))?;
    let name_end = cursor
        .checked_add(name_length)
        .ok_or_else(|| import_error("FBX node name overflow"))?;
    let name = std::str::from_utf8(
        bytes
            .get(*cursor..name_end)
            .ok_or_else(|| import_error("FBX node name is truncated"))?,
    )
    .map_err(|_| import_error("FBX node name is not UTF-8"))?
    .to_owned();
    *cursor = name_end;
    let properties_end = cursor
        .checked_add(property_len)
        .ok_or_else(|| import_error("FBX property list overflow"))?;
    if properties_end > bytes.len() {
        return Err(import_error("FBX property list is truncated"));
    }
    if end_offset > bytes.len() || property_count > property_len || property_len > MAX_FILE_BYTES {
        return Err(import_error(
            "FBX node offsets or property counts exceed their bounds",
        ));
    }
    let mut args = Vec::with_capacity(property_count);
    let mut children = Vec::new();
    while *cursor < properties_end {
        let kind = *bytes
            .get(*cursor)
            .ok_or_else(|| import_error("FBX property type is missing"))?
            as char;
        let argument = decode_property(bytes, cursor, properties_end)?;
        if matches!(kind, 'd' | 'f' | 'i' | 'l' | 'b') {
            let array_args = if argument.value.is_empty() {
                Vec::new()
            } else {
                argument
                    .value
                    .split(',')
                    .map(|value| Arg {
                        value: value.to_owned(),
                        is_string: false,
                    })
                    .collect()
            };
            children.push(Node {
                name: "a".to_owned(),
                args: array_args,
                children: Vec::new(),
            });
        } else {
            args.push(argument);
        }
    }
    if *cursor != properties_end || args.len() + children.len() != property_count {
        return Err(import_error(
            "FBX property list length does not match its count",
        ));
    }
    let null_len = if wide { 25 } else { 13 };
    while *cursor < end_offset {
        if bytes.len().saturating_sub(*cursor) < null_len {
            return Err(import_error("FBX child table is truncated"));
        }
        if bytes[*cursor..*cursor + null_len]
            .iter()
            .all(|byte| *byte == 0)
        {
            *cursor += null_len;
            break;
        }
        if let Some(child) = decode_node(bytes, cursor, version)? {
            children.push(child);
        } else {
            break;
        }
    }
    if *cursor != end_offset {
        return Err(import_error("FBX node end offset is inconsistent"));
    }
    if end_offset <= start {
        return Err(import_error("FBX node has an invalid end offset"));
    }
    Ok(Some(Node {
        name,
        args,
        children,
    }))
}

fn decode_property(bytes: &[u8], cursor: &mut usize, end: usize) -> Result<Arg> {
    let kind = *bytes
        .get(*cursor)
        .ok_or_else(|| import_error("FBX property type is missing"))? as char;
    *cursor += 1;
    let value = match kind {
        'S' | 'R' => {
            let len = super::read_u32(bytes, *cursor, READ_ERRORS)? as usize;
            *cursor = cursor
                .checked_add(4)
                .ok_or_else(|| import_error("FBX string cursor overflow"))?;
            let string_end = cursor
                .checked_add(len)
                .ok_or_else(|| import_error("FBX string length overflow"))?;
            if string_end > end {
                return Err(import_error("FBX string property is truncated"));
            }
            let value = std::str::from_utf8(&bytes[*cursor..string_end])
                .map_err(|_| import_error("FBX string property is not UTF-8"))?
                .to_owned();
            *cursor = string_end;
            Arg {
                value,
                is_string: true,
            }
        }
        'Y' => {
            let value = i16::from_le_bytes(read_array(bytes, cursor)?);
            Arg {
                value: value.to_string(),
                is_string: false,
            }
        }
        'C' => {
            let value = *bytes
                .get(*cursor)
                .ok_or_else(|| import_error("FBX boolean is truncated"))?;
            *cursor += 1;
            Arg {
                value: i64::from(value != 0).to_string(),
                is_string: false,
            }
        }
        'I' => Arg {
            value: i32::from_le_bytes(read_array(bytes, cursor)?).to_string(),
            is_string: false,
        },
        'F' => Arg {
            value: format_number(f32::from_le_bytes(read_array(bytes, cursor)?) as f64),
            is_string: false,
        },
        'D' => Arg {
            value: format_number(f64::from_le_bytes(read_array(bytes, cursor)?)),
            is_string: false,
        },
        'L' => Arg {
            value: i64::from_le_bytes(read_array(bytes, cursor)?).to_string(),
            is_string: false,
        },
        'd' | 'f' | 'i' | 'l' | 'b' => {
            let count = super::read_u32(bytes, *cursor, READ_ERRORS)? as usize;
            let encoding = super::read_u32(bytes, *cursor + 4, READ_ERRORS)?;
            let compressed_len = super::read_u32(bytes, *cursor + 8, READ_ERRORS)? as usize;
            *cursor = cursor
                .checked_add(12)
                .ok_or_else(|| import_error("FBX array header overflow"))?;
            let compressed_end = cursor
                .checked_add(compressed_len)
                .ok_or_else(|| import_error("FBX array length overflow"))?;
            if compressed_end > end {
                return Err(import_error("FBX compressed array is truncated"));
            }
            let component_len = array_component_len(kind);
            let expected_len = count
                .checked_mul(component_len)
                .ok_or_else(|| import_error("FBX array byte length overflow"))?;
            if expected_len > MAX_FILE_BYTES {
                return Err(PotError::new(
                    ErrorCode::LimitExceeded,
                    "FBX array exceeds the size limit",
                ));
            }
            let raw = if encoding == 0 {
                if compressed_len != expected_len {
                    return Err(import_error(
                        "FBX uncompressed array has the wrong byte length",
                    ));
                }
                bytes[*cursor..compressed_end].to_vec()
            } else if encoding == 1 {
                let mut decoder = ZlibDecoder::new(&bytes[*cursor..compressed_end]).take(
                    u64::try_from(expected_len)
                        .map_err(|_| import_error("FBX array length exceeds the limit"))?
                        .saturating_add(1),
                );
                let mut output = Vec::with_capacity(expected_len);
                decoder.read_to_end(&mut output).map_err(|error| {
                    PotError::new(
                        ErrorCode::ImportFailed,
                        format!("FBX array decompression failed: {error}"),
                    )
                })?;
                if output.len() != expected_len {
                    return Err(import_error(
                        "FBX compressed array has the wrong byte length",
                    ));
                }
                output
            } else {
                return Err(import_error("FBX array uses an unknown compression mode"));
            };
            *cursor = compressed_end;
            let values = decode_array_values(&raw, kind)?;
            Arg {
                value: values,
                is_string: false,
            }
        }
        _ => return Err(import_error("FBX binary property type is unsupported")),
    };
    Ok(value)
}

fn decode_array_values(bytes: &[u8], kind: char) -> Result<String> {
    let width = array_component_len(kind);
    let mut values = Vec::with_capacity(bytes.len() / width);
    for chunk in bytes.chunks_exact(width) {
        let value = match kind {
            'd' => format_number(f64::from_le_bytes(
                chunk
                    .try_into()
                    .map_err(|_| import_error("invalid FBX double array"))?,
            )),
            'f' => format_number(f32::from_le_bytes(
                chunk
                    .try_into()
                    .map_err(|_| import_error("invalid FBX float array"))?,
            ) as f64),
            'i' => i32::from_le_bytes(
                chunk
                    .try_into()
                    .map_err(|_| import_error("invalid FBX integer array"))?,
            )
            .to_string(),
            'l' => i64::from_le_bytes(
                chunk
                    .try_into()
                    .map_err(|_| import_error("invalid FBX long array"))?,
            )
            .to_string(),
            'b' => i64::from(chunk[0] != 0).to_string(),
            _ => return Err(import_error("FBX array type is unsupported")),
        };
        values.push(value);
    }
    Ok(values.join(","))
}

fn render_ascii(nodes: &[Node]) -> String {
    let mut output = String::from("; FBX binary decoded as ASCII\n");
    for node in nodes {
        render_node(&mut output, node, 0);
    }
    output
}

fn render_node(output: &mut String, node: &Node, depth: usize) {
    let indent = "\t".repeat(depth);
    output.push_str(&indent);
    output.push_str(&node.name);
    output.push(':');
    if node.name == "a" {
        output.push(' ');
        output.push_str(
            &node
                .args
                .iter()
                .map(|arg| arg.value.as_str())
                .collect::<Vec<_>>()
                .join(","),
        );
        output.push('\n');
        return;
    }
    for (index, arg) in node.args.iter().enumerate() {
        if index == 0 {
            output.push(' ');
        } else {
            output.push(',');
        }
        let rendered_value = if index == 1 {
            arg.value.split_once("\0\x01").map(|(name, class)| {
                let ascii_class = match class {
                    "AnimStack" => "AnimationStack",
                    "AnimLayer" => "AnimationLayer",
                    "AnimCurveNode" => "AnimationCurveNode",
                    "AnimCurve" => "AnimationCurve",
                    class => class,
                };
                format!("{ascii_class}::{name}")
            })
        } else {
            None
        };
        let value = rendered_value.as_deref().unwrap_or(&arg.value);
        if arg.is_string {
            output.push('"');
            output.push_str(&escape_ascii(value));
            output.push('"');
        } else {
            output.push_str(value);
        }
    }
    if has_array_child(node) {
        output.push_str(" {\n");
        for child in &node.children {
            render_node(output, child, depth + 1);
        }
        output.push_str(&indent);
        output.push_str("}\n");
    } else if node.children.is_empty() {
        output.push('\n');
    } else {
        output.push_str(" {\n");
        for child in &node.children {
            render_node(output, child, depth + 1);
        }
        output.push_str(&indent);
        output.push_str("}\n");
    }
}

fn has_array_child(node: &Node) -> bool {
    node.children.iter().any(|child| child.name == "a")
}

fn escape_ascii(value: &str) -> String {
    value.replace('\\', "\\\\").replace('"', "\\\"")
}

fn array_component_len(kind: char) -> usize {
    match kind {
        'd' | 'l' => 8,
        'f' | 'i' => 4,
        'b' => 1,
        _ => 0,
    }
}

fn parse_i64(value: &str) -> Result<i64> {
    value
        .parse::<i64>()
        .map_err(|_| export_error("ASCII FBX integer property is invalid"))
}

fn parse_f64(value: &str) -> Result<f64> {
    let number = value
        .parse::<f64>()
        .map_err(|_| export_error("ASCII FBX numeric property is invalid"))?;
    if number.is_finite() {
        Ok(number)
    } else {
        Err(export_error("ASCII FBX number is non-finite"))
    }
}

fn write_u32_at(output: &mut [u8], offset: usize, value: u32) {
    output[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
}

fn write_u64_at(output: &mut [u8], offset: usize, value: u64) {
    output[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
}

fn read_array<const N: usize>(bytes: &[u8], cursor: &mut usize) -> Result<[u8; N]> {
    let end = cursor
        .checked_add(N)
        .ok_or_else(|| import_error("FBX cursor overflow"))?;
    let value = bytes
        .get(*cursor..end)
        .ok_or_else(|| import_error("FBX property is truncated"))?;
    *cursor = end;
    value
        .try_into()
        .map_err(|_| import_error("FBX property has an invalid size"))
}

fn format_number(value: f64) -> String {
    if value == 0.0 {
        "0".to_owned()
    } else {
        format!("{value:.17}")
            .trim_end_matches('0')
            .trim_end_matches('.')
            .to_owned()
    }
}

fn limit_error(message: &str) -> PotError {
    PotError::new(ErrorCode::LimitExceeded, message)
}

fn export_error(message: &str) -> PotError {
    PotError::new(ErrorCode::ExportFailed, message)
}

#[cfg(test)]
mod tests {
    use super::{Arg, Node, decode_binary, encode_binary, import_error};

    #[test]
    fn compressed_array_properties_round_trip_in_74_and_75() -> crate::error::Result<()> {
        let vertices = Node {
            name: "Vertices".to_owned(),
            args: vec![Arg {
                value: "2".to_owned(),
                is_string: false,
            }],
            children: vec![Node {
                name: "a".to_owned(),
                args: vec![
                    Arg {
                        value: "1.25".to_owned(),
                        is_string: false,
                    },
                    Arg {
                        value: "-2.5".to_owned(),
                        is_string: false,
                    },
                ],
                children: Vec::new(),
            }],
        };
        for version in [7400, 7500] {
            let bytes = encode_binary(std::slice::from_ref(&vertices), version)?;
            let (nodes, decoded_version) = decode_binary(&bytes)?;
            assert_eq!(decoded_version, version);
            let decoded = nodes
                .first()
                .ok_or_else(|| import_error("FBX node is missing"))?;
            assert!(decoded.args.is_empty());
            let array = decoded
                .children
                .first()
                .ok_or_else(|| import_error("FBX array property is missing"))?;
            assert_eq!(
                array
                    .args
                    .iter()
                    .map(|argument| argument.value.as_str())
                    .collect::<Vec<_>>(),
                vec!["1.25", "-2.5"]
            );
        }
        Ok(())
    }
}
