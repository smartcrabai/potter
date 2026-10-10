use std::{
    collections::{BTreeMap, BTreeSet},
    fmt::Write as _,
    fs,
    path::Path,
};

use crate::{
    error::{ErrorCode, PotError, Result},
    eval::Snapshot,
    model::SceneDoc,
};
use serde_json::json;

use super::{ImportedGraph, import_error};

const IDENT: &[u8; 8] = b"PXR-USDC";
const VERSION_MAJOR: u8 = 0;
const VERSION_MINOR: u8 = 4;
const VERSION_PATCH: u8 = 0;
const BOOTSTRAP_SIZE: usize = 88;
const ARRAY_BIT: u64 = 1 << 63;
const INLINE_BIT: u64 = 1 << 62;
const COMPRESSED_BIT: u64 = 1 << 61;

/// Write a USD crate containing the scene subset emitted by the USDA exporter.
pub(crate) fn export(
    doc: &SceneDoc,
    snapshot: &Snapshot,
    _blender: Option<&Path>,
) -> Result<Vec<u8>> {
    let text = super::usd::export_usda(doc, snapshot)?;
    let prims = parse_usda_subset(std::str::from_utf8(&text).map_err(|_| {
        PotError::new(
            ErrorCode::ExportFailed,
            "USD exporter produced non-UTF-8 text",
        )
    })?)?;
    encode_crate(&prims)
}

/// Read a native USD crate and pass its decoded USDA representation to the shared importer.
pub(crate) fn import(
    file: &Path,
    scene_id: String,
    _blender: Option<&Path>,
) -> Result<ImportedGraph> {
    if !file.is_file() {
        return Err(PotError::new(
            ErrorCode::FileNotFound,
            "USD crate file does not exist",
        ));
    }
    let bytes = fs::read(file).map_err(|error| PotError::io(&error))?;
    let layer = decode_crate(&bytes)?;
    let path = std::env::temp_dir().join(format!("potter-{}.usda", uuid::Uuid::new_v4()));
    let result = (|| {
        fs::write(&path, layer.as_bytes()).map_err(|error| PotError::io(&error))?;
        let mut imported = super::usd::import_usda(&path, scene_id)?;
        imported.source = json!({ "format": "usdc", "path": file.display().to_string() });
        Ok(imported)
    })();
    let _ = fs::remove_file(path);
    result
}

#[derive(Clone, Debug)]
struct UsdPrim {
    kind: String,
    name: String,
    properties: Vec<UsdProperty>,
    children: Vec<UsdPrim>,
}

#[derive(Clone, Debug)]
struct UsdProperty {
    name: String,
    type_name: String,
    custom: bool,
    uniform: bool,
    interpolation: Option<String>,
    time_samples: Option<(Vec<f64>, Vec<ValueData>)>,
    raw_value: String,
}

fn parse_usda_subset(text: &str) -> Result<Vec<UsdPrim>> {
    if !text.starts_with("#usda 1.0") {
        return Err(PotError::new(
            ErrorCode::ExportFailed,
            "USD exporter did not produce USDA 1.0",
        ));
    }
    let mut roots = Vec::<UsdPrim>::new();
    let mut stack = Vec::<usize>::new();
    for line in text.lines() {
        let line = line.trim();
        if let Some(rest) = line.strip_prefix("def ") {
            let split = rest.find(char::is_whitespace).ok_or_else(|| {
                PotError::new(ErrorCode::ExportFailed, "malformed USDA prim declaration")
            })?;
            let kind = rest[..split].to_owned();
            let quoted = rest[split..].trim();
            let name = quoted
                .strip_prefix('"')
                .and_then(|value| value.find('"').map(|end| &value[..end]))
                .ok_or_else(|| PotError::new(ErrorCode::ExportFailed, "malformed USDA prim name"))?
                .to_owned();
            let prim = UsdPrim {
                kind,
                name,
                properties: Vec::new(),
                children: Vec::new(),
            };
            if stack.is_empty() {
                roots.push(prim);
                stack.push(roots.len() - 1);
            } else {
                let parent = prim_at_mut(&mut roots, &stack)?;
                parent.children.push(prim);
                stack.push(parent.children.len() - 1);
            }
            continue;
        }
        if line == "}" {
            if !stack.is_empty() {
                stack.pop();
            }
            continue;
        }
        let Some((left, right)) = line.split_once('=') else {
            continue;
        };
        if stack.is_empty() {
            continue;
        }
        let current = prim_at_mut(&mut roots, &stack)?;
        let mut words = left.split_whitespace().collect::<Vec<_>>();
        if words.is_empty() || words[0] == "rel" {
            continue;
        }
        let custom = words.first() == Some(&"custom");
        let uniform = words.first() == Some(&"uniform");
        if custom || uniform {
            words.remove(0);
        }
        if words.len() < 2 {
            continue;
        }
        let name = words[1].to_owned();
        let type_name = words[0].to_owned();
        if name.ends_with(".connect") {
            continue;
        }
        let raw = right.trim();
        if let Some(attribute_name) = name.strip_suffix(".timeSamples") {
            let (times, samples) = parse_time_samples(&type_name, raw)?;
            if let Some(index) = current
                .properties
                .iter()
                .position(|property| property.name == attribute_name)
            {
                current.properties[index].time_samples = Some((times, samples));
            } else {
                current.properties.push(UsdProperty {
                    name: attribute_name.to_owned(),
                    type_name,
                    custom,
                    uniform,
                    interpolation: None,
                    time_samples: Some((times, samples)),
                    raw_value: String::new(),
                });
            }
            continue;
        }
        let interpolation = raw
            .split_once("(interpolation = ")
            .and_then(|(_, suffix)| parse_quoted_strings(suffix).ok())
            .and_then(|values| values.into_iter().next());
        let Some(raw_value) = raw.split(" (interpolation =").next() else {
            continue;
        };
        current.properties.push(UsdProperty {
            name,
            type_name,
            custom,
            uniform,
            interpolation,
            time_samples: None,
            raw_value: raw_value.trim().to_owned(),
        });
    }
    if roots.is_empty() {
        return Err(PotError::new(
            ErrorCode::ExportFailed,
            "USDA scene has no prims",
        ));
    }
    Ok(roots)
}

fn prim_at_mut<'a>(roots: &'a mut [UsdPrim], path: &[usize]) -> Result<&'a mut UsdPrim> {
    let (first, rest) = path
        .split_first()
        .ok_or_else(|| PotError::new(ErrorCode::ExportFailed, "USDA prim nesting is malformed"))?;
    let mut current = roots
        .get_mut(*first)
        .ok_or_else(|| PotError::new(ErrorCode::ExportFailed, "USDA prim nesting is malformed"))?;
    for index in rest {
        current = current.children.get_mut(*index).ok_or_else(|| {
            PotError::new(ErrorCode::ExportFailed, "USDA prim nesting is malformed")
        })?;
    }
    Ok(current)
}

#[derive(Clone, Debug)]
enum ValueData {
    Bool(bool),
    I32(i32),
    I64(i64),
    Enum {
        code: u8,
        value: i32,
    },
    F32(f32),
    F64(f64),
    String(String),
    Token(String),
    NumericArray {
        type_code: u8,
        bytes: Vec<u8>,
    },
    StringArray(Vec<String>),
    TokenArray(Vec<String>),
    TokenVector(Vec<String>),
    DoubleVector(Vec<f64>),
    Vec3d([f64; 3]),
    Vec3f([f32; 3]),
    Vec2f([f32; 2]),
    Quatd([f64; 4]),
    Matrix4d([f64; 16]),
    TimeSamples {
        times: Vec<f64>,
        values: Vec<ValueData>,
    },
}

#[derive(Clone, Debug)]
struct CrateField {
    name: String,
    value: ValueData,
}

#[derive(Clone, Debug)]
struct CrateSpec {
    path: String,
    spec_type: u32,
    fields: Vec<CrateField>,
}

fn encode_crate(prims: &[UsdPrim]) -> Result<Vec<u8>> {
    let mut specs = vec![CrateSpec {
        path: "/".to_owned(),
        spec_type: 7,
        fields: vec![
            CrateField {
                name: "defaultPrim".to_owned(),
                value: ValueData::Token("Scene".to_owned()),
            },
            CrateField {
                name: "metersPerUnit".to_owned(),
                value: ValueData::F64(1.0),
            },
            CrateField {
                name: "upAxis".to_owned(),
                value: ValueData::Token("Z".to_owned()),
            },
            CrateField {
                name: "primChildren".to_owned(),
                value: ValueData::TokenVector(prims.iter().map(|prim| prim.name.clone()).collect()),
            },
            CrateField {
                name: "properties".to_owned(),
                value: ValueData::TokenVector(Vec::new()),
            },
        ],
    }];
    specs[0]
        .fields
        .sort_by(|left, right| left.name.cmp(&right.name));
    append_prim_specs(prims, "/", &mut specs)?;
    specs.sort_by(|left, right| left.path.cmp(&right.path));
    let mut tokens = BTreeSet::<String>::new();
    tokens.insert(String::new());
    for spec in &specs {
        for part in path_parts(&spec.path) {
            tokens.insert(part);
        }
        for field in &spec.fields {
            tokens.insert(field.name.clone());
            collect_value_tokens(&field.value, &mut tokens);
            collect_value_strings(&field.value, &mut tokens);
        }
    }
    let tokens = tokens.into_iter().collect::<Vec<_>>();
    let token_indexes = tokens
        .iter()
        .enumerate()
        .map(|(index, token)| (token.clone(), index as u32))
        .collect::<BTreeMap<_, _>>();
    let mut strings = BTreeSet::<String>::new();
    for spec in &specs {
        for field in &spec.fields {
            collect_value_strings(&field.value, &mut strings);
        }
    }
    let strings = strings.into_iter().collect::<Vec<_>>();
    let string_indexes = strings
        .iter()
        .enumerate()
        .map(|(index, value)| (value.clone(), index as u32))
        .collect::<BTreeMap<_, _>>();
    let string_tokens = strings
        .iter()
        .map(|value| token_indexes[value])
        .collect::<Vec<_>>();
    let mut file = vec![0_u8; BOOTSTRAP_SIZE];
    let mut field_records = Vec::<(u32, u64)>::new();
    let mut field_set_values = Vec::<u32>::new();
    let mut encoded_specs = Vec::<(u32, u32, u32)>::new();
    let path_strings = make_paths(&specs);
    let path_indexes = path_strings
        .iter()
        .enumerate()
        .map(|(index, path)| (path.clone(), index as u32))
        .collect::<BTreeMap<_, _>>();
    for spec in &specs {
        let field_set_start = u32::try_from(field_set_values.len())
            .map_err(|_| export_error("too many USD fields"))?;
        for field in &spec.fields {
            let field_token = *token_indexes
                .get(&field.name)
                .ok_or_else(|| export_error("missing token index"))?;
            let value_rep = encode_value(&field.value, &token_indexes, &string_indexes, &mut file)?;
            field_records.push((field_token, value_rep));
            let field_index = u32::try_from(field_records.len() - 1)
                .map_err(|_| export_error("too many USD fields"))?;
            field_set_values.push(field_index);
        }
        field_set_values.push(u32::MAX);
        let path_index = *path_indexes
            .get(&spec.path)
            .ok_or_else(|| export_error("missing USD path index"))?;
        encoded_specs.push((path_index, field_set_start, spec.spec_type));
    }
    let mut sections = Vec::<Section>::new();
    let mut token_data = Vec::new();
    for token in &tokens {
        token_data.extend_from_slice(token.as_bytes());
        token_data.push(0);
    }
    let compressed_tokens = fast_compress(&token_data)?;
    let mut token_section = Vec::new();
    put_u64(&mut token_section, tokens.len() as u64);
    put_u64(&mut token_section, token_data.len() as u64);
    put_u64(&mut token_section, compressed_tokens.len() as u64);
    token_section.extend_from_slice(&compressed_tokens);
    append_section(&mut file, &mut sections, "TOKENS", &token_section);
    let mut string_section = Vec::new();
    put_u64(&mut string_section, string_tokens.len() as u64);
    for token in string_tokens {
        put_u32(&mut string_section, token);
    }
    append_section(&mut file, &mut sections, "STRINGS", &string_section);
    let mut fields_section = Vec::new();
    put_u64(&mut fields_section, field_records.len() as u64);
    let field_tokens = field_records
        .iter()
        .map(|(token, _)| i32::try_from(*token).map_err(|_| export_error("too many USD tokens")))
        .collect::<Result<Vec<_>>>()?;
    append_compressed_ints(&mut fields_section, &field_tokens)?;
    let mut reps = Vec::with_capacity(field_records.len() * 8);
    for (_, rep) in &field_records {
        put_u64(&mut reps, *rep);
    }
    let compressed_reps = fast_compress(&reps)?;
    put_u64(&mut fields_section, compressed_reps.len() as u64);
    fields_section.extend_from_slice(&compressed_reps);
    append_section(&mut file, &mut sections, "FIELDS", &fields_section);
    let mut fieldsets_section = Vec::new();
    put_u64(&mut fieldsets_section, field_set_values.len() as u64);
    let fieldsets = field_set_values
        .iter()
        .map(|value| {
            if *value == u32::MAX {
                Ok(-1)
            } else {
                i32::try_from(*value).map_err(|_| export_error("too many USD fields"))
            }
        })
        .collect::<Result<Vec<_>>>()?;
    append_compressed_ints(&mut fieldsets_section, &fieldsets)?;
    append_section(&mut file, &mut sections, "FIELDSETS", &fieldsets_section);
    let paths_section = encode_paths(&path_strings, &token_indexes)?;
    append_section(&mut file, &mut sections, "PATHS", &paths_section);
    let mut specs_section = Vec::new();
    put_u64(&mut specs_section, encoded_specs.len() as u64);
    let path_indexes = encoded_specs
        .iter()
        .map(|(path, _, _)| i32::try_from(*path).map_err(|_| export_error("too many USD paths")))
        .collect::<Result<Vec<_>>>()?;
    let spec_fieldsets = encoded_specs
        .iter()
        .map(|(_, fieldset, _)| {
            i32::try_from(*fieldset).map_err(|_| export_error("too many USD fields"))
        })
        .collect::<Result<Vec<_>>>()?;
    let spec_types = encoded_specs
        .iter()
        .map(|(_, _, spec_type)| {
            i32::try_from(*spec_type).map_err(|_| export_error("invalid USD spec type"))
        })
        .collect::<Result<Vec<_>>>()?;
    append_compressed_ints(&mut specs_section, &path_indexes)?;
    append_compressed_ints(&mut specs_section, &spec_fieldsets)?;
    append_compressed_ints(&mut specs_section, &spec_types)?;
    append_section(&mut file, &mut sections, "SPECS", &specs_section);
    let toc_offset = file.len();
    put_u64(&mut file, sections.len() as u64);
    for section in sections {
        let mut name = [0_u8; 16];
        name[..section.name.len()].copy_from_slice(section.name.as_bytes());
        file.extend_from_slice(&name);
        put_i64(&mut file, section.start as i64);
        put_i64(&mut file, section.size as i64);
    }
    file[..8].copy_from_slice(IDENT);
    file[8] = VERSION_MAJOR;
    file[9] = VERSION_MINOR;
    file[10] = VERSION_PATCH;
    file[16..24].copy_from_slice(&(toc_offset as i64).to_le_bytes());
    Ok(file)
}

fn append_prim_specs(prims: &[UsdPrim], parent: &str, specs: &mut Vec<CrateSpec>) -> Result<()> {
    for prim in prims {
        let path = if parent == "/" {
            format!("/{0}", prim.name)
        } else {
            format!("{parent}/{0}", prim.name)
        };
        let mut fields = vec![
            CrateField {
                name: "primChildren".to_owned(),
                value: ValueData::TokenVector(
                    prim.children
                        .iter()
                        .map(|child| child.name.clone())
                        .collect(),
                ),
            },
            CrateField {
                name: "properties".to_owned(),
                value: ValueData::TokenVector(
                    prim.properties
                        .iter()
                        .map(|property| property.name.clone())
                        .collect(),
                ),
            },
            CrateField {
                name: "specifier".to_owned(),
                value: ValueData::Enum { code: 42, value: 0 },
            },
            CrateField {
                name: "typeName".to_owned(),
                value: ValueData::Token(prim.kind.clone()),
            },
        ];
        fields.sort_by(|a, b| a.name.cmp(&b.name));
        specs.push(CrateSpec {
            path: path.clone(),
            spec_type: 6,
            fields,
        });
        for property in &prim.properties {
            let default_value = if property.raw_value.is_empty() {
                None
            } else {
                parse_value(&property.type_name, &property.raw_value)?
            };
            if default_value.is_none() && property.time_samples.is_none() {
                continue;
            }
            let attribute_path = format!("{path}.{}", property.name);
            let mut fields = vec![
                CrateField {
                    name: "custom".to_owned(),
                    value: ValueData::Bool(property.custom),
                },
                CrateField {
                    name: "typeName".to_owned(),
                    value: ValueData::Token(property.type_name.clone()),
                },
                CrateField {
                    name: "variability".to_owned(),
                    value: ValueData::Enum {
                        code: 44,
                        value: i32::from(property.uniform),
                    },
                },
            ];
            if let Some(value) = default_value {
                fields.push(CrateField {
                    name: "default".to_owned(),
                    value,
                });
            }
            if let Some((times, values)) = &property.time_samples {
                fields.push(CrateField {
                    name: "timeSamples".to_owned(),
                    value: ValueData::TimeSamples {
                        times: times.clone(),
                        values: values.clone(),
                    },
                });
            }
            if let Some(interpolation) = &property.interpolation {
                fields.push(CrateField {
                    name: "interpolation".to_owned(),
                    value: ValueData::Token(interpolation.clone()),
                });
            }
            fields.sort_by(|a, b| a.name.cmp(&b.name));
            specs.push(CrateSpec {
                path: attribute_path,
                spec_type: 1,
                fields,
            });
        }
        append_prim_specs(&prim.children, &path, specs)?;
    }
    Ok(())
}

fn parse_value(type_name: &str, raw: &str) -> Result<Option<ValueData>> {
    let is_array = type_name.ends_with("[]");
    let scalar_type = type_name.strip_suffix("[]").unwrap_or(type_name);
    if is_array {
        match scalar_type {
            "string" => return Ok(Some(ValueData::StringArray(parse_quoted_strings(raw)?))),
            "token" => return Ok(Some(ValueData::TokenArray(parse_quoted_strings(raw)?))),
            "int" => {
                return Ok(Some(ValueData::NumericArray {
                    type_code: 3,
                    bytes: numbers(raw)
                        .into_iter()
                        .map(parse_i32)
                        .collect::<Result<Vec<_>>>()?
                        .into_iter()
                        .flatten()
                        .collect(),
                }));
            }
            "double" => {
                return Ok(Some(ValueData::NumericArray {
                    type_code: 9,
                    bytes: numbers(raw)
                        .into_iter()
                        .flat_map(f64::to_le_bytes)
                        .collect(),
                }));
            }
            "float" => {
                return Ok(Some(ValueData::NumericArray {
                    type_code: 8,
                    bytes: numbers(raw)
                        .into_iter()
                        .flat_map(|value| (value as f32).to_le_bytes())
                        .collect(),
                }));
            }
            "point3f" | "normal3f" | "color3f" => {
                return Ok(Some(ValueData::NumericArray {
                    type_code: 24,
                    bytes: numbers(raw)
                        .into_iter()
                        .flat_map(|value| (value as f32).to_le_bytes())
                        .collect(),
                }));
            }
            "texCoord2f" => {
                return Ok(Some(ValueData::NumericArray {
                    type_code: 20,
                    bytes: numbers(raw)
                        .into_iter()
                        .flat_map(|value| (value as f32).to_le_bytes())
                        .collect(),
                }));
            }
            "double3" => {
                return Ok(Some(ValueData::NumericArray {
                    type_code: 23,
                    bytes: numbers(raw)
                        .into_iter()
                        .flat_map(f64::to_le_bytes)
                        .collect(),
                }));
            }
            _ => {
                return Err(unsupported_export(&format!(
                    "USD array type `{type_name}` is not supported"
                )));
            }
        }
    }
    let parsed = match scalar_type {
        "string" => ValueData::String(
            parse_quoted_strings(raw)?
                .into_iter()
                .next()
                .ok_or_else(|| export_error("invalid USDA string"))?,
        ),
        "token" => ValueData::Token(
            parse_quoted_strings(raw)?
                .into_iter()
                .next()
                .unwrap_or_else(|| raw.trim().to_owned()),
        ),
        "bool" => ValueData::Bool(raw.trim().starts_with("true")),
        "int" => ValueData::I32(parse_i32_value(
            numbers(raw)
                .first()
                .copied()
                .ok_or_else(|| export_error("invalid USDA integer"))?,
        )?),
        "int64" => ValueData::I64(
            numbers(raw)
                .first()
                .copied()
                .ok_or_else(|| export_error("invalid USDA integer"))? as i64,
        ),
        "float" => ValueData::F32(
            numbers(raw)
                .first()
                .copied()
                .ok_or_else(|| export_error("invalid USDA float"))? as f32,
        ),
        "double" => ValueData::F64(
            numbers(raw)
                .first()
                .copied()
                .ok_or_else(|| export_error("invalid USDA double"))?,
        ),
        "double3" => ValueData::Vec3d(number_tuple::<3>(raw)?),
        "float3" | "point3f" | "normal3f" | "color3f" => {
            ValueData::Vec3f(number_tuple::<3>(raw)?.map(|v| v as f32))
        }
        "texCoord2f" | "float2" => ValueData::Vec2f(number_tuple::<2>(raw)?.map(|v| v as f32)),
        "quatd" => ValueData::Quatd(number_tuple::<4>(raw)?),
        "matrix4d" => ValueData::Matrix4d(number_tuple::<16>(raw)?),
        _ => {
            return Err(unsupported_export(&format!(
                "USD attribute type `{type_name}` is not supported"
            )));
        }
    };
    Ok(Some(parsed))
}
fn parse_time_samples(type_name: &str, raw: &str) -> Result<(Vec<f64>, Vec<ValueData>)> {
    let width = match type_name {
        "double3" | "float3" | "point3f" | "normal3f" | "color3f" => 3,
        "quatd" => 4,
        "float2" | "texCoord2f" => 2,
        "float" | "double" | "int" => 1,
        _ => {
            return Err(unsupported_export(&format!(
                "USD timeSamples type `{type_name}` is not supported"
            )));
        }
    };
    let values = numbers(raw);
    let stride = width + 1;
    if !values.len().is_multiple_of(stride) {
        return Err(export_error(
            "USDA timeSamples have an invalid component count",
        ));
    }
    let mut times = Vec::with_capacity(values.len() / stride);
    let mut samples = Vec::with_capacity(values.len() / stride);
    for sample in values.chunks_exact(stride) {
        times.push(sample[0]);
        let components = &sample[1..];
        let value = match type_name {
            "double3" => ValueData::Vec3d([components[0], components[1], components[2]]),
            "float3" | "point3f" | "normal3f" | "color3f" => ValueData::Vec3f([
                components[0] as f32,
                components[1] as f32,
                components[2] as f32,
            ]),
            "quatd" => {
                ValueData::Quatd([components[0], components[1], components[2], components[3]])
            }
            "float2" | "texCoord2f" => {
                ValueData::Vec2f([components[0] as f32, components[1] as f32])
            }
            "double" => ValueData::F64(components[0]),
            "float" => ValueData::F32(components[0] as f32),
            "int" => ValueData::I32(parse_i32_value(components[0])?),
            _ => {
                return Err(unsupported_export(&format!(
                    "USD timeSamples type `{type_name}` is not supported"
                )));
            }
        };
        samples.push(value);
    }
    Ok((times, samples))
}

fn numbers(input: &str) -> Vec<f64> {
    let bytes = input.as_bytes();
    let mut result = Vec::new();
    let mut start = None;
    for (index, byte) in bytes.iter().enumerate() {
        let is_number = byte.is_ascii_digit()
            || *byte == b'+'
            || *byte == b'-'
            || *byte == b'.'
            || *byte == b'e'
            || *byte == b'E';
        if is_number {
            if start.is_none() {
                start = Some(index);
            }
        } else if let Some(begin) = start.take()
            && let Ok(value) = input[begin..index].parse::<f64>()
        {
            result.push(value);
        }
    }
    if let Some(begin) = start
        && let Ok(value) = input[begin..].parse::<f64>()
    {
        result.push(value);
    }
    result
}

fn number_tuple<const N: usize>(raw: &str) -> Result<[f64; N]> {
    let values = numbers(raw);
    let array: [f64; N] = values
        .try_into()
        .map_err(|_| export_error("USDA vector has the wrong component count"))?;
    Ok(array)
}

fn parse_i32_value(value: f64) -> Result<i32> {
    if !value.is_finite()
        || value.fract() != 0.0
        || value < i32::MIN as f64
        || value > i32::MAX as f64
    {
        return Err(export_error("USDA integer value is out of range"));
    }
    Ok(value as i32)
}

fn parse_i32(value: f64) -> Result<Vec<u8>> {
    Ok(parse_i32_value(value)?.to_le_bytes().to_vec())
}

fn parse_quoted_strings(raw: &str) -> Result<Vec<String>> {
    let mut strings = Vec::new();
    let chars = raw.char_indices().collect::<Vec<_>>();
    let mut index = 0;
    while index < chars.len() {
        let (position, character) = chars[index];
        if character != '"' {
            index += 1;
            continue;
        }
        let mut value = String::new();
        index += 1;
        let mut closed = false;
        while index < chars.len() {
            let (_, current) = chars[index];
            if current == '"' {
                closed = true;
                index += 1;
                break;
            }
            if current == '\\' {
                index += 1;
                let Some((_, escaped)) = chars.get(index).copied() else {
                    return Err(export_error("invalid USDA quoted string"));
                };
                value.push(match escaped {
                    'n' => '\n',
                    'r' => '\r',
                    't' => '\t',
                    other => other,
                });
                index += 1;
            } else {
                value.push(current);
                index += 1;
            }
        }
        if !closed {
            return Err(export_error("unterminated USDA string"));
        }
        strings.push(value);
        let _ = position;
    }
    Ok(strings)
}

fn collect_value_tokens(value: &ValueData, tokens: &mut BTreeSet<String>) {
    match value {
        ValueData::Token(value) => {
            tokens.insert(value.clone());
        }
        ValueData::TokenArray(values) | ValueData::TokenVector(values) => {
            tokens.extend(values.iter().cloned());
        }
        ValueData::TimeSamples { values, .. } => {
            for value in values {
                collect_value_tokens(value, tokens);
            }
        }
        _ => {}
    }
}

fn collect_value_strings(value: &ValueData, strings: &mut BTreeSet<String>) {
    match value {
        ValueData::String(value) => {
            strings.insert(value.clone());
        }
        ValueData::StringArray(values) => strings.extend(values.iter().cloned()),
        ValueData::TimeSamples { values, .. } => {
            for value in values {
                collect_value_strings(value, strings);
            }
        }
        _ => {}
    }
}

fn encode_value(
    value: &ValueData,
    tokens: &BTreeMap<String, u32>,
    strings: &BTreeMap<String, u32>,
    file: &mut Vec<u8>,
) -> Result<u64> {
    if let ValueData::TimeSamples { times, values } = value {
        if times.len() != values.len() {
            return Err(export_error(
                "USD timeSamples have mismatched time and value counts",
            ));
        }
        let payload = file.len();
        let times_pointer = file.len();
        put_i64(file, 0);
        let times_rep = encode_value(
            &ValueData::DoubleVector(times.clone()),
            tokens,
            strings,
            file,
        )?;
        let times_offset = i64::try_from(file.len() - times_pointer)
            .map_err(|_| export_error("USD timeSamples offset exceeds crate limits"))?;
        file[times_pointer..times_pointer + 8].copy_from_slice(&times_offset.to_le_bytes());
        put_u64(file, times_rep);
        let values_pointer = file.len();
        put_i64(file, 0);
        let value_reps = values
            .iter()
            .map(|sample| encode_value(sample, tokens, strings, file))
            .collect::<Result<Vec<_>>>()?;
        let values_offset = i64::try_from(file.len() - values_pointer)
            .map_err(|_| export_error("USD timeSamples offset exceeds crate limits"))?;
        file[values_pointer..values_pointer + 8].copy_from_slice(&values_offset.to_le_bytes());
        put_u64(file, value_reps.len() as u64);
        for rep in value_reps {
            put_u64(file, rep);
        }
        return Ok((46_u64 << 48) | payload as u64);
    }
    let (code, array, payload, bytes, inline) = match value {
        ValueData::Bool(value) => (1, false, u64::from(*value), Vec::new(), true),
        ValueData::I32(value) => (
            3,
            false,
            u32::from_le_bytes(value.to_le_bytes()) as u64,
            Vec::new(),
            true,
        ),
        ValueData::I64(value) => (5, false, 0, value.to_le_bytes().to_vec(), false),
        ValueData::Enum { code, value } => (
            *code,
            false,
            u32::from_le_bytes(value.to_le_bytes()) as u64,
            Vec::new(),
            true,
        ),
        ValueData::F32(value) => (
            8,
            false,
            u32::from_le_bytes(value.to_le_bytes()) as u64,
            Vec::new(),
            true,
        ),
        ValueData::F64(value) => (9, false, 0, value.to_le_bytes().to_vec(), false),
        ValueData::String(value) => (
            10,
            false,
            u64::from(
                *strings
                    .get(value)
                    .ok_or_else(|| export_error("missing USD string index"))?,
            ),
            Vec::new(),
            true,
        ),
        ValueData::Token(value) => (
            11,
            false,
            u64::from(
                *tokens
                    .get(value)
                    .ok_or_else(|| export_error("missing USD token index"))?,
            ),
            Vec::new(),
            true,
        ),
        ValueData::Vec3d(values) => (
            23,
            false,
            0,
            values
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect(),
            false,
        ),
        ValueData::Vec3f(values) => (
            24,
            false,
            0,
            values
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect(),
            false,
        ),
        ValueData::Vec2f(values) => (
            20,
            false,
            0,
            values
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect(),
            false,
        ),
        ValueData::Quatd(values) => (
            16,
            false,
            0,
            values
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect(),
            false,
        ),
        ValueData::Matrix4d(values) => (
            15,
            false,
            0,
            values
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect(),
            false,
        ),
        ValueData::NumericArray { type_code, bytes } => (*type_code, true, 0, bytes.clone(), false),
        ValueData::DoubleVector(values) => {
            let mut bytes = Vec::with_capacity(8 + values.len() * 8);
            put_u64(&mut bytes, values.len() as u64);
            for value in values {
                bytes.extend_from_slice(&value.to_le_bytes());
            }
            (48, false, 0, bytes, false)
        }
        ValueData::StringArray(values) => (
            10,
            true,
            0,
            values
                .iter()
                .map(|value| {
                    strings
                        .get(value)
                        .copied()
                        .ok_or_else(|| export_error("missing USD string index"))
                })
                .collect::<Result<Vec<_>>>()?
                .into_iter()
                .flat_map(u32::to_le_bytes)
                .collect(),
            false,
        ),
        ValueData::TokenArray(values) => (
            11,
            true,
            0,
            values
                .iter()
                .map(|value| {
                    tokens
                        .get(value)
                        .copied()
                        .ok_or_else(|| export_error("missing USD token index"))
                })
                .collect::<Result<Vec<_>>>()?
                .into_iter()
                .flat_map(u32::to_le_bytes)
                .collect(),
            false,
        ),
        ValueData::TokenVector(values) => {
            let mut bytes = Vec::with_capacity(8 + values.len() * 4);
            put_u64(&mut bytes, values.len() as u64);
            for value in values {
                put_u32(
                    &mut bytes,
                    *tokens
                        .get(value)
                        .ok_or_else(|| export_error("missing USD token index"))?,
                );
            }
            (41, false, 0, bytes, false)
        }
        ValueData::TimeSamples { .. } => {
            return Err(export_error("nested USD timeSamples are unsupported"));
        }
    };
    let mut rep_payload = payload;
    if array {
        if !bytes.is_empty() {
            align8(file);
            rep_payload = file.len() as u64;
            put_u32(file, 1);
            let element_size = match code {
                1 | 2 => 1,
                3 | 4 | 8 | 10 | 11 => 4,
                24 => 12,
                23 => 24,
                20 | 9 => 8,
                _ => return Err(export_error("unsupported USD array element type")),
            };
            let count = bytes.len() / element_size;
            put_u32(
                file,
                u32::try_from(count).map_err(|_| export_error("USD array exceeds crate limits"))?,
            );
            file.extend_from_slice(&bytes);
        }
    } else if !inline {
        rep_payload = file.len() as u64;
        file.extend_from_slice(&bytes);
    }
    Ok(((code as u64) << 48)
        | (if array { ARRAY_BIT } else { 0 })
        | (if inline { INLINE_BIT } else { 0 })
        | (rep_payload & ((1_u64 << 48) - 1)))
}

#[derive(Clone, Debug)]
struct Section {
    name: String,
    start: usize,
    size: usize,
}

fn append_section(file: &mut Vec<u8>, sections: &mut Vec<Section>, name: &str, data: &[u8]) {
    let start = file.len();
    file.extend_from_slice(data);
    sections.push(Section {
        name: name.to_owned(),
        start,
        size: data.len(),
    });
}

fn make_paths(specs: &[CrateSpec]) -> Vec<String> {
    let mut paths = BTreeSet::new();
    paths.insert("/".to_owned());
    for spec in specs {
        paths.insert(spec.path.clone());
    }
    paths.into_iter().collect()
}

fn path_parts(path: &str) -> Vec<String> {
    if path == "/" {
        return vec![String::new()];
    }
    let name = path.rsplit(['/', '.']).next().unwrap_or_default();
    vec![name.to_owned()]
}

fn path_parent(path: &str) -> Option<String> {
    if path == "/" {
        return None;
    }
    let slash = path.rfind('/').unwrap_or(0);
    if let Some(dot) = path.rfind('.')
        && dot > slash
    {
        return Some(path[..dot].to_owned());
    }
    if slash == 0 {
        Some("/".to_owned())
    } else {
        Some(path[..slash].to_owned())
    }
}

fn encode_paths(paths: &[String], tokens: &BTreeMap<String, u32>) -> Result<Vec<u8>> {
    let mut children = BTreeMap::<String, Vec<String>>::new();
    for path in paths.iter().filter(|path| path.as_str() != "/") {
        let parent = path_parent(path).ok_or_else(|| export_error("invalid USD path"))?;
        children.entry(parent).or_default().push(path.clone());
    }
    for siblings in children.values_mut() {
        siblings.sort();
    }
    let indexes = paths
        .iter()
        .enumerate()
        .map(|(index, path)| (path.clone(), index as u32))
        .collect::<BTreeMap<_, _>>();
    let mut path_indexes = Vec::new();
    let mut element_tokens = Vec::new();
    let mut jumps = Vec::new();
    collect_compressed_paths(
        &["/".to_owned()],
        &children,
        &indexes,
        tokens,
        true,
        &mut path_indexes,
        &mut element_tokens,
        &mut jumps,
    )?;
    let mut bytes = Vec::new();
    put_u64(&mut bytes, paths.len() as u64);
    put_u64(&mut bytes, path_indexes.len() as u64);
    append_compressed_ints(&mut bytes, &path_indexes)?;
    append_compressed_ints(&mut bytes, &element_tokens)?;
    append_compressed_ints(&mut bytes, &jumps)?;
    Ok(bytes)
}

fn collect_compressed_paths(
    siblings: &[String],
    children: &BTreeMap<String, Vec<String>>,
    path_indexes: &BTreeMap<String, u32>,
    tokens: &BTreeMap<String, u32>,
    root_level: bool,
    encoded_paths: &mut Vec<i32>,
    element_tokens: &mut Vec<i32>,
    jumps: &mut Vec<i32>,
) -> Result<()> {
    for (position, path) in siblings.iter().enumerate() {
        let this_index = encoded_paths.len();
        let path_index = *path_indexes
            .get(path)
            .ok_or_else(|| export_error("missing USD path index"))?;
        encoded_paths
            .push(i32::try_from(path_index).map_err(|_| export_error("too many USD paths"))?);
        let property = !root_level
            && path
                .rfind('.')
                .is_some_and(|dot| dot > path.rfind('/').unwrap_or(0));
        let element = if root_level {
            String::new()
        } else {
            path.rsplit(['/', '.'])
                .next()
                .unwrap_or_default()
                .to_owned()
        };
        let token_index = *tokens
            .get(&element)
            .ok_or_else(|| export_error("missing path token"))?;
        let token_index =
            i32::try_from(token_index).map_err(|_| export_error("too many USD tokens"))?;
        element_tokens.push(if property { -token_index } else { token_index });
        jumps.push(0);
        let nested = children.get(path).map_or(&[][..], Vec::as_slice);
        let has_child = !nested.is_empty();
        let has_sibling = position + 1 < siblings.len();
        if has_child {
            collect_compressed_paths(
                nested,
                children,
                path_indexes,
                tokens,
                false,
                encoded_paths,
                element_tokens,
                jumps,
            )?;
        }
        jumps[this_index] = if has_child && has_sibling {
            i32::try_from(encoded_paths.len() - this_index)
                .map_err(|_| export_error("too many USD paths"))?
        } else if has_sibling {
            0
        } else if has_child {
            -1
        } else {
            -2
        };
    }
    Ok(())
}

fn append_compressed_ints(output: &mut Vec<u8>, values: &[i32]) -> Result<()> {
    let integer_data = encode_integer_data(values);
    let compressed = fast_compress(&integer_data)?;
    put_u64(output, compressed.len() as u64);
    output.extend_from_slice(&compressed);
    Ok(())
}

fn encode_integer_data(values: &[i32]) -> Vec<u8> {
    if values.is_empty() {
        return Vec::new();
    }
    let mut deltas = Vec::with_capacity(values.len());
    let mut counts = BTreeMap::<i32, usize>::new();
    let mut previous = 0_i32;
    for value in values {
        let delta = value.wrapping_sub(previous);
        *counts.entry(delta).or_default() += 1;
        deltas.push(delta);
        previous = *value;
    }
    let common = counts
        .into_iter()
        .max_by_key(|(value, count)| (*count, *value))
        .map(|(value, _)| value)
        .unwrap_or_default();
    let mut codes = vec![0_u8; (values.len() * 2).div_ceil(8)];
    let mut variable = Vec::new();
    for (index, delta) in deltas.iter().enumerate() {
        let code = if *delta == common {
            0
        } else if let Ok(value) = i8::try_from(*delta) {
            variable.push(value as u8);
            1
        } else if let Ok(value) = i16::try_from(*delta) {
            variable.extend_from_slice(&value.to_le_bytes());
            2
        } else {
            variable.extend_from_slice(&delta.to_le_bytes());
            3
        };
        codes[index / 4] |= code << ((index % 4) * 2);
    }
    let mut encoded = Vec::with_capacity(4 + codes.len() + variable.len());
    put_i32(&mut encoded, common);
    encoded.extend_from_slice(&codes);
    encoded.extend_from_slice(&variable);
    encoded
}

fn fast_compress(input: &[u8]) -> Result<Vec<u8>> {
    if input.len() > 0x7e00_0000 {
        return Err(export_error("USD crate section exceeds LZ4 block limit"));
    }
    let mut output = vec![0_u8];
    let literal_prefix = input.len().min(15);
    output.push((literal_prefix as u8) << 4);
    if input.len() >= 15 {
        let mut remaining = input.len() - 15;
        while remaining >= 255 {
            output.push(255);
            remaining -= 255;
        }
        output.push(remaining as u8);
    }
    output.extend_from_slice(input);
    Ok(output)
}

fn decode_crate(bytes: &[u8]) -> Result<String> {
    if bytes.len() < BOOTSTRAP_SIZE || bytes.get(..8) != Some(IDENT.as_slice()) {
        return Err(import_error("input is not a USD crate file"));
    }
    let major = bytes[8];
    let minor = bytes[9];
    if major != 0 || minor > 11 {
        return Err(PotError::new(
            ErrorCode::UnsupportedVersion,
            "USD crate version is unsupported",
        ));
    }
    let toc_offset = usize::try_from(read_i64(bytes, 16)?)
        .map_err(|_| import_error("invalid crate TOC offset"))?;
    let mut toc = Cursor::at(bytes, toc_offset)?;
    let section_count = toc.read_u64()? as usize;
    let mut sections = BTreeMap::<String, (usize, &[u8])>::new();
    let mut ranges = Vec::new();
    for _ in 0..section_count {
        let name = toc.read_bytes(16)?;
        let end = name
            .iter()
            .position(|value| *value == 0)
            .unwrap_or(name.len());
        let name = std::str::from_utf8(&name[..end])
            .map_err(|_| import_error("invalid crate section name"))?
            .to_owned();
        let start = usize::try_from(toc.read_i64()?)
            .map_err(|_| import_error("invalid crate section offset"))?;
        let size = usize::try_from(toc.read_i64()?)
            .map_err(|_| import_error("invalid crate section size"))?;
        let section_end = start
            .checked_add(size)
            .ok_or_else(|| import_error("crate section size overflow"))?;
        if start < BOOTSTRAP_SIZE || section_end > toc_offset {
            return Err(import_error("crate section overlaps the header or TOC"));
        }
        let section = bytes
            .get(start..section_end)
            .ok_or_else(|| import_error("crate section exceeds file bounds"))?;
        if sections.insert(name, (start, section)).is_some() {
            return Err(import_error("crate TOC contains a duplicate section"));
        }
        ranges.push((start, section_end));
    }
    ranges.sort_unstable();
    if ranges.windows(2).any(|pair| pair[0].1 > pair[1].0) {
        return Err(import_error("crate sections overlap"));
    }
    let tokens = decode_tokens(required_section(&sections, "TOKENS")?, minor)?;
    let strings = decode_strings(required_section(&sections, "STRINGS")?)?;
    let fields = decode_fields(required_section(&sections, "FIELDS")?, minor)?;
    let field_sets = decode_fieldsets(required_section(&sections, "FIELDSETS")?, minor)?;
    let (paths_start, paths_section) = sections
        .get("PATHS")
        .copied()
        .ok_or_else(|| import_error("USD crate is missing a required structural section"))?;
    let paths = decode_paths(paths_section, paths_start, minor, &tokens)?;
    let specs = decode_specs(required_section(&sections, "SPECS")?, minor)?;
    layer_from_specs(
        &specs,
        &paths,
        &field_sets,
        &fields,
        &tokens,
        &strings,
        bytes,
        minor,
    )
}

fn required_section<'a>(
    sections: &'a BTreeMap<String, (usize, &'a [u8])>,
    name: &str,
) -> Result<&'a [u8]> {
    sections
        .get(name)
        .map(|(_, section)| *section)
        .ok_or_else(|| import_error("USD crate is missing a required structural section"))
}
fn decode_tokens(section: &[u8], minor: u8) -> Result<Vec<String>> {
    let mut cursor = Cursor::new(section);
    let count = cursor.read_u64()? as usize;
    let data = if minor < 4 {
        let size = cursor.read_u64()? as usize;
        cursor.read_bytes(size)?.to_vec()
    } else {
        let uncompressed = cursor.read_u64()? as usize;
        let compressed = cursor.read_u64()? as usize;
        fast_decompress(cursor.read_bytes(compressed)?, uncompressed)?
    };
    let mut result = Vec::with_capacity(count);
    let mut start = 0;
    for _ in 0..count {
        let end = data
            .get(start..)
            .and_then(|tail| tail.iter().position(|byte| *byte == 0))
            .map(|offset| start + offset)
            .ok_or_else(|| import_error("invalid null-terminated USD crate token"))?;
        result.push(
            std::str::from_utf8(&data[start..end])
                .map_err(|_| import_error("crate token is not UTF-8"))?
                .to_owned(),
        );
        start = end + 1;
    }
    Ok(result)
}

fn decode_strings(section: &[u8]) -> Result<Vec<u32>> {
    let mut cursor = Cursor::new(section);
    let count = cursor.read_u64()? as usize;
    (0..count).map(|_| cursor.read_u32()).collect()
}

fn decode_fields(section: &[u8], minor: u8) -> Result<Vec<(u32, u64)>> {
    let mut cursor = Cursor::new(section);
    if minor < 4 {
        let count = cursor.read_u64()? as usize;
        let mut fields = Vec::with_capacity(count);
        for _ in 0..count {
            let _padding = cursor.read_u32()?;
            let token = cursor.read_u32()?;
            fields.push((token, cursor.read_u64()?));
        }
        return Ok(fields);
    }
    let count = cursor.read_u64()? as usize;
    let token_indexes = read_compressed_ints(&mut cursor, count)?;
    let compressed_size = cursor.read_u64()? as usize;
    let compressed = cursor.read_bytes(compressed_size)?;
    let reps = fast_decompress(
        compressed,
        count
            .checked_mul(8)
            .ok_or_else(|| import_error("too many crate fields"))?,
    )?;
    let mut fields = Vec::with_capacity(count);
    for index in 0..count {
        let rep = u64::from_le_bytes(
            reps[index * 8..index * 8 + 8]
                .try_into()
                .map_err(|_| import_error("invalid field rep"))?,
        );
        fields.push((token_indexes[index], rep));
    }
    Ok(fields)
}

fn decode_fieldsets(section: &[u8], minor: u8) -> Result<Vec<u32>> {
    let mut cursor = Cursor::new(section);
    if minor < 4 {
        let count = cursor.read_u64()? as usize;
        return (0..count).map(|_| cursor.read_u32()).collect();
    }
    let count = cursor.read_u64()? as usize;
    read_compressed_ints(&mut cursor, count)
}

fn decode_specs(section: &[u8], minor: u8) -> Result<Vec<(u32, u32, u32)>> {
    let mut cursor = Cursor::new(section);
    if minor < 4 {
        let count = cursor.read_u64()? as usize;
        let mut result = Vec::with_capacity(count);
        for _ in 0..count {
            result.push((cursor.read_u32()?, cursor.read_u32()?, cursor.read_u32()?));
        }
        return Ok(result);
    }
    let count = cursor.read_u64()? as usize;
    let paths = read_compressed_ints(&mut cursor, count)?;
    let fieldsets = read_compressed_ints(&mut cursor, count)?;
    let types = read_compressed_ints(&mut cursor, count)?;
    Ok((0..count)
        .map(|i| (paths[i], fieldsets[i], types[i]))
        .collect())
}

fn decode_paths(
    section: &[u8],
    section_start: usize,
    minor: u8,
    tokens: &[String],
) -> Result<Vec<String>> {
    let mut cursor = Cursor::new(section);
    let count = cursor.read_u64()? as usize;
    let mut paths = vec![String::new(); count];
    if minor < 4 {
        decode_path_tree(
            section,
            section_start,
            cursor.position,
            "/",
            tokens,
            &mut paths,
            true,
        )?;
    } else {
        let encoded_count = cursor.read_u64()? as usize;
        let indexes = read_compressed_ints(&mut cursor, encoded_count)?;
        let elements = read_compressed_i32s(&mut cursor, encoded_count)?;
        let jumps = read_compressed_i32s(&mut cursor, encoded_count)?;
        decode_compressed_path_level(0, "/", &indexes, &elements, &jumps, tokens, &mut paths)?;
    }
    Ok(paths)
}

fn decode_path_tree(
    section: &[u8],
    section_start: usize,
    start: usize,
    parent: &str,
    tokens: &[String],
    paths: &mut [String],
    root: bool,
) -> Result<usize> {
    let mut position = start;
    loop {
        let mut cursor = Cursor::at(section, position)?;
        let index = cursor.read_u32()? as usize;
        let token_index = cursor.read_u32()? as usize;
        let bits = cursor.read_u8()?;
        let _padding = cursor.read_bytes(3)?;
        let name = tokens
            .get(token_index)
            .ok_or_else(|| import_error("crate path token index is out of range"))?;
        let property = bits & 4 != 0;
        let path = if root {
            "/".to_owned()
        } else if parent == "/" {
            format!("/{name}")
        } else if property {
            format!("{parent}.{name}")
        } else {
            format!("{parent}/{name}")
        };
        let target = paths
            .get_mut(index)
            .ok_or_else(|| import_error("crate path index is out of range"))?;
        target.clone_from(&path);
        let has_child = bits & 1 != 0;
        let has_sibling = bits & 2 != 0;
        let sibling = if has_child && has_sibling {
            let absolute = usize::try_from(cursor.read_i64()?)
                .map_err(|_| import_error("invalid path sibling offset"))?;
            Some(
                absolute
                    .checked_sub(section_start)
                    .ok_or_else(|| import_error("path sibling offset precedes its section"))?,
            )
        } else {
            None
        };
        position = cursor.position;
        if has_child {
            position = decode_path_tree(
                section,
                section_start,
                position,
                &path,
                tokens,
                paths,
                false,
            )?;
        }
        if let Some(sibling) = sibling {
            position = sibling;
        }
        if !has_sibling {
            return Ok(position);
        }
    }
}

fn decode_compressed_path_level(
    mut index: usize,
    parent: &str,
    indexes: &[u32],
    elements: &[i32],
    jumps: &[i32],
    tokens: &[String],
    paths: &mut [String],
) -> Result<usize> {
    loop {
        if index >= indexes.len() {
            return Err(import_error("truncated compressed crate paths"));
        }
        let element_index = elements[index].unsigned_abs() as usize;
        let element = tokens
            .get(element_index)
            .ok_or_else(|| import_error("crate path token index is out of range"))?;
        let path = if index == 0 {
            "/".to_owned()
        } else if elements[index] < 0 {
            format!("{parent}.{element}")
        } else if parent == "/" {
            format!("/{element}")
        } else {
            format!("{parent}/{element}")
        };
        let path_index = indexes[index] as usize;
        paths
            .get_mut(path_index)
            .ok_or_else(|| import_error("crate path index is out of range"))?
            .clone_from(&path);
        let jump = jumps[index];
        let has_child = jump > 0 || jump == -1;
        let has_sibling = jump >= 0;
        let after_child = if has_child {
            decode_compressed_path_level(index + 1, &path, indexes, elements, jumps, tokens, paths)?
        } else {
            index + 1
        };
        if has_sibling {
            index = if has_child {
                index
                    .checked_add(jump as usize)
                    .ok_or_else(|| import_error("crate path jump overflow"))?
            } else {
                index + 1
            };
        } else {
            return Ok(after_child);
        }
    }
}

fn read_compressed_ints(cursor: &mut Cursor<'_>, count: usize) -> Result<Vec<u32>> {
    let size = cursor.read_u64()? as usize;
    let compressed = cursor.read_bytes(size)?;
    let decoded = fast_decompress_limited(compressed, integer_working_size(count)?)?;
    decode_integer_data(&decoded, count)
        .map(|values| values.into_iter().map(|value| value as u32).collect())
}

fn read_compressed_i32s(cursor: &mut Cursor<'_>, count: usize) -> Result<Vec<i32>> {
    let size = cursor.read_u64()? as usize;
    let compressed = cursor.read_bytes(size)?;
    let decoded = fast_decompress_limited(compressed, integer_working_size(count)?)?;
    decode_integer_data(&decoded, count)
}

fn integer_working_size(count: usize) -> Result<usize> {
    if count == 0 {
        return Ok(0);
    }
    count
        .checked_mul(4)
        .and_then(|value| value.checked_add(count.saturating_mul(2).saturating_add(7) / 8 + 4))
        .ok_or_else(|| import_error("compressed integer list is too large"))
}

fn decode_integer_data(data: &[u8], count: usize) -> Result<Vec<i32>> {
    if count == 0 {
        return Ok(Vec::new());
    }
    if data.len() < 4 + (count * 2).div_ceil(8) {
        return Err(import_error("truncated crate integer coding"));
    }
    let mut position = 0;
    let common = take_i32(data, &mut position)?;
    let codes_start = position;
    let values_start = codes_start + (count * 2).div_ceil(8);
    let mut values_position = values_start;
    let mut previous = 0_i32;
    let mut output = Vec::with_capacity(count);
    for index in 0..count {
        let code = (data[codes_start + index / 4] >> ((index % 4) * 2)) & 3;
        let difference = match code {
            0 => common,
            1 => take_i8(data, &mut values_position)? as i32,
            2 => take_i16(data, &mut values_position)? as i32,
            _ => take_i32(data, &mut values_position)?,
        };
        previous = previous.wrapping_add(difference);
        output.push(previous);
    }
    Ok(output)
}

fn fast_decompress(input: &[u8], expected: usize) -> Result<Vec<u8>> {
    let output = fast_decompress_limited(input, expected)?;
    if output.len() != expected {
        return Err(import_error(
            "decompressed crate section has an unexpected size",
        ));
    }
    Ok(output)
}

fn fast_decompress_limited(input: &[u8], maximum: usize) -> Result<Vec<u8>> {
    let mut cursor = Cursor::new(input);
    let chunks = cursor.read_u8()?;
    let mut output = Vec::with_capacity(maximum);
    if chunks == 0 {
        let block = cursor.read_bytes(cursor.remaining())?;
        lz4_block(block, maximum, &mut output)?;
    } else {
        for _ in 0..chunks {
            let size = usize::try_from(cursor.read_i32()?)
                .map_err(|_| import_error("invalid LZ4 crate chunk size"))?;
            let block = cursor.read_bytes(size)?;
            let mut chunk = Vec::new();
            lz4_block(block, 0x7e00_0000, &mut chunk)?;
            if output
                .len()
                .checked_add(chunk.len())
                .is_none_or(|length| length > maximum)
            {
                return Err(import_error("LZ4 output exceeds expected size"));
            }
            output.extend_from_slice(&chunk);
        }
    }
    Ok(output)
}

fn lz4_block(input: &[u8], max_output: usize, output: &mut Vec<u8>) -> Result<()> {
    let mut position = 0;
    while position < input.len() {
        let token = input[position];
        position += 1;
        let mut literals = usize::from(token >> 4);
        if literals == 15 {
            literals = literals
                .checked_add(read_lz4_length(input, &mut position)?)
                .ok_or_else(|| import_error("LZ4 literal length overflow"))?;
        }
        let literal_end = position
            .checked_add(literals)
            .ok_or_else(|| import_error("LZ4 literal length overflow"))?;
        let literal_bytes = input
            .get(position..literal_end)
            .ok_or_else(|| import_error("truncated LZ4 literals"))?;
        if output
            .len()
            .checked_add(literals)
            .is_none_or(|length| length > max_output)
        {
            return Err(import_error("LZ4 output exceeds expected size"));
        }
        output.extend_from_slice(literal_bytes);
        position = literal_end;
        if position == input.len() {
            break;
        }
        let offset_bytes = input
            .get(position..position + 2)
            .ok_or_else(|| import_error("truncated LZ4 match offset"))?;
        let offset = usize::from(u16::from_le_bytes([offset_bytes[0], offset_bytes[1]]));
        position += 2;
        if offset == 0 || offset > output.len() {
            return Err(import_error("invalid LZ4 match offset"));
        }
        let mut length = usize::from(token & 15) + 4;
        if token & 15 == 15 {
            length = length
                .checked_add(read_lz4_length(input, &mut position)?)
                .ok_or_else(|| import_error("LZ4 match length overflow"))?;
        }
        if output
            .len()
            .checked_add(length)
            .is_none_or(|size| size > max_output)
        {
            return Err(import_error("LZ4 output exceeds expected size"));
        }
        for _ in 0..length {
            let byte = output[output.len() - offset];
            output.push(byte);
        }
    }
    Ok(())
}

fn read_lz4_length(input: &[u8], position: &mut usize) -> Result<usize> {
    let mut length = 0_usize;
    loop {
        let byte = *input
            .get(*position)
            .ok_or_else(|| import_error("truncated LZ4 extended length"))?;
        *position += 1;
        length = length
            .checked_add(usize::from(byte))
            .ok_or_else(|| import_error("LZ4 length overflow"))?;
        if byte != 255 {
            return Ok(length);
        }
    }
}

fn layer_from_specs(
    specs: &[(u32, u32, u32)],
    paths: &[String],
    fieldsets: &[u32],
    fields: &[(u32, u64)],
    tokens: &[String],
    strings: &[u32],
    bytes: &[u8],
    minor: u8,
) -> Result<String> {
    let mut layer = String::from("#usda 1.0\n\n");
    let mut by_path = BTreeMap::<String, BTreeMap<String, DecodedValue>>::new();
    let mut seen_paths = BTreeSet::new();
    for (path_index, fieldset, spec_type) in specs {
        if !matches!(*spec_type, 1 | 6 | 7) {
            return Err(unsupported_import(
                "USD crate relationship, mapper, or variant specs are unsupported",
            ));
        }
        let path = paths
            .get(*path_index as usize)
            .ok_or_else(|| import_error("crate spec references an invalid path"))?;
        if (*spec_type == 7) != (path == "/") {
            return Err(import_error("crate pseudo-root spec has an invalid path"));
        }
        if !seen_paths.insert(path.clone()) {
            return Err(import_error(
                "USD crate contains duplicate specs for one path",
            ));
        }
        let mut field_index = *fieldset as usize;
        let mut values = BTreeMap::new();
        let mut terminated = false;
        while let Some(index) = fieldsets.get(field_index).copied() {
            field_index += 1;
            if index == u32::MAX {
                terminated = true;
                break;
            }
            let (token_index, rep) = fields
                .get(index as usize)
                .ok_or_else(|| import_error("crate fieldset references an invalid field"))?;
            let name = tokens
                .get(*token_index as usize)
                .ok_or_else(|| import_error("crate field references an invalid token"))?;
            values.insert(
                name.clone(),
                decode_value(*rep, tokens, strings, bytes, minor)?,
            );
        }
        if !terminated {
            return Err(import_error("USD crate fieldset is missing its terminator"));
        }
        by_path.insert(path.clone(), values);
    }
    if let Some(root) = by_path.get("/") {
        let mut metadata = Vec::new();
        for name in [
            "defaultPrim",
            "metersPerUnit",
            "upAxis",
            "timeCodesPerSecond",
            "startTimeCode",
            "endTimeCode",
        ] {
            if let Some(value) = root.get(name) {
                metadata.push(format!(
                    "    {name} = {}",
                    value.to_usda(if name == "defaultPrim" || name == "upAxis" {
                        "string"
                    } else {
                        "double"
                    })?
                ));
            }
        }
        if !metadata.is_empty() {
            layer.push_str("(\n");
            for entry in metadata {
                layer.push_str(&entry);
                layer.push('\n');
            }
            layer.push_str(")\n\n");
        }
    }
    let mut prims = BTreeMap::<String, OutPrim>::new();
    for (path_index, _, spec_type) in specs {
        if *spec_type != 6 {
            continue;
        }
        let path = paths
            .get(*path_index as usize)
            .ok_or_else(|| import_error("crate spec references an invalid path"))?;
        let values = by_path
            .get(path)
            .ok_or_else(|| import_error("crate prim fields are missing"))?;
        let name = path.rsplit('/').next().unwrap_or_default().to_owned();
        let kind = values
            .get("typeName")
            .and_then(DecodedValue::as_text)
            .ok_or_else(|| import_error("USD prim spec has no typeName token"))?
            .to_owned();
        prims.insert(
            path.clone(),
            OutPrim {
                name,
                kind,
                properties: Vec::new(),
                children: Vec::new(),
            },
        );
    }
    for (path_index, _, spec_type) in specs {
        if *spec_type != 1 {
            continue;
        }
        let path = paths
            .get(*path_index as usize)
            .ok_or_else(|| import_error("crate spec references an invalid path"))?;
        let Some((parent_path, name)) = path.rsplit_once('.') else {
            return Err(import_error("USD attribute spec has a malformed path"));
        };
        let Some(prim) = prims.get_mut(parent_path) else {
            continue;
        };
        let values = by_path
            .get(path)
            .ok_or_else(|| import_error("crate attribute fields are missing"))?;
        let type_name = values
            .get("typeName")
            .and_then(DecodedValue::as_text)
            .ok_or_else(|| import_error("USD attribute has no typeName"))?;
        let default = values.get("default");
        let custom = matches!(values.get("custom"), Some(DecodedValue::Bool(true)));
        let uniform = matches!(values.get("variability"), Some(DecodedValue::I32(1)));
        let interpolation = values.get("interpolation").and_then(DecodedValue::as_text);
        let metadata = interpolation
            .map(|value| format!(" (interpolation = \"{}\")", escape(value)))
            .unwrap_or_default();
        if let Some(value) = default {
            let rendered = value.to_usda(type_name)?;
            prim.properties.push(format!(
                "{}{} {} = {}{}",
                if custom {
                    "custom "
                } else if uniform {
                    "uniform "
                } else {
                    ""
                },
                type_name,
                name,
                rendered,
                metadata
            ));
        }
        if let Some(samples) = values.get("timeSamples") {
            prim.properties.push(format!(
                "{type_name} {name}.timeSamples = {}",
                samples.to_usda(type_name)?
            ));
        }
    }
    for prim in prims.values_mut() {
        prim.properties.sort();
    }
    let paths_sorted = prims.keys().cloned().collect::<Vec<_>>();
    for path in paths_sorted.iter().rev() {
        if let Some(parent_path) = path_parent(path)
            && parent_path != "/"
            && let Some(child) = prims.remove(path)
        {
            if let Some(parent) = prims.get_mut(&parent_path) {
                parent.children.push(child);
            } else {
                prims.insert(path.clone(), child);
            }
        }
    }
    for prim in prims.values_mut() {
        prim.children
            .sort_by(|left, right| left.name.cmp(&right.name));
    }
    for prim in prims.values().filter(|prim| prim.name == "Scene") {
        write_prim(&mut layer, prim, 0);
    }
    if !layer.contains("def Xform \"Scene\"") {
        return Err(import_error("USD crate has no /Scene prim"));
    }
    Ok(layer)
}

#[derive(Clone, Debug)]
enum DecodedValue {
    Bool(bool),
    I32(i32),
    I64(i64),
    F32(f32),
    F64(f64),
    Text(String),
    TextArray(Vec<String>),
    Numbers {
        type_code: u8,
        values: Vec<f64>,
    },
    TimeSamples {
        times: Vec<f64>,
        values: Vec<DecodedValue>,
    },
}

impl DecodedValue {
    fn as_text(&self) -> Option<&str> {
        if let Self::Text(value) = self {
            Some(value)
        } else {
            None
        }
    }
    fn to_usda(&self, type_name: &str) -> Result<String> {
        match self {
            Self::Bool(value) => Ok(value.to_string()),
            Self::I32(value) => Ok(value.to_string()),
            Self::I64(value) => Ok(value.to_string()),
            Self::F32(value) => Ok(format_number(f64::from(*value))),
            Self::F64(value) => Ok(format_number(*value)),
            Self::Text(value) => Ok(format!("\"{}\"", escape(value))),
            Self::TextArray(values) => Ok(format!(
                "[{}]",
                values
                    .iter()
                    .map(|value| format!("\"{}\"", escape(value)))
                    .collect::<Vec<_>>()
                    .join(", ")
            )),
            Self::Numbers { type_code, values } => {
                if type_name.ends_with("[]") {
                    let width = match type_code {
                        20 => 2,
                        23 | 24 | 26 => 3,
                        27 => 4,
                        _ => 1,
                    };
                    let items = values
                        .chunks(width)
                        .map(|chunk| {
                            if width == 1 {
                                format_number(chunk[0])
                            } else {
                                format!(
                                    "({})",
                                    chunk
                                        .iter()
                                        .map(|value| format_number(*value))
                                        .collect::<Vec<_>>()
                                        .join(", ")
                                )
                            }
                        })
                        .collect::<Vec<_>>();
                    Ok(format!("[{}]", items.join(", ")))
                } else if *type_code == 16 && values.len() == 4 {
                    Ok(format!(
                        "({}, ({}, {}, {}))",
                        format_number(values[0]),
                        format_number(values[1]),
                        format_number(values[2]),
                        format_number(values[3])
                    ))
                } else if *type_code == 15 && values.len() == 16 {
                    let rows = values
                        .chunks(4)
                        .map(|row| {
                            format!(
                                "({}, {}, {}, {})",
                                format_number(row[0]),
                                format_number(row[1]),
                                format_number(row[2]),
                                format_number(row[3])
                            )
                        })
                        .collect::<Vec<_>>();
                    Ok(format!("({})", rows.join(", ")))
                } else if values.len() == 1 {
                    Ok(format_number(values[0]))
                } else {
                    Ok(format!(
                        "({})",
                        values
                            .iter()
                            .map(|value| format_number(*value))
                            .collect::<Vec<_>>()
                            .join(", ")
                    ))
                }
            }
            Self::TimeSamples { times, values } => {
                if times.len() != values.len() {
                    return Err(import_error(
                        "USD timeSamples count does not match its value reps",
                    ));
                }
                let samples = times
                    .iter()
                    .zip(values)
                    .map(|(time, value)| {
                        Ok(format!(
                            "{}: {}",
                            format_number(*time),
                            value.to_usda(type_name)?
                        ))
                    })
                    .collect::<Result<Vec<_>>>()?;
                Ok(format!("{{ {} }}", samples.join(", ")))
            }
        }
    }
}

fn decode_value(
    rep: u64,
    tokens: &[String],
    strings: &[u32],
    bytes: &[u8],
    minor: u8,
) -> Result<DecodedValue> {
    let code = ((rep >> 48) & 0xff) as u8;
    let array = rep & ARRAY_BIT != 0;
    let inlined = rep & INLINE_BIT != 0;
    let compressed = rep & COMPRESSED_BIT != 0;
    let payload = (rep & ((1_u64 << 48) - 1)) as usize;
    if array {
        if payload == 0 {
            return Ok(DecodedValue::Numbers {
                type_code: code,
                values: Vec::new(),
            });
        }
        let mut cursor = Cursor::at(bytes, payload)?;
        if minor < 5 {
            let _shape = cursor.read_u32()?;
        }
        let count = if minor < 7 {
            cursor.read_u32()? as usize
        } else {
            cursor.read_u64()? as usize
        };
        if code == 10 || code == 11 {
            let mut values = Vec::with_capacity(count);
            for _ in 0..count {
                let index = cursor.read_u32()? as usize;
                let token_index = if code == 10 {
                    *strings
                        .get(index)
                        .ok_or_else(|| import_error("string array index is invalid"))?
                        as usize
                } else {
                    index
                };
                values.push(
                    tokens
                        .get(token_index)
                        .ok_or_else(|| import_error("string array token is invalid"))?
                        .clone(),
                );
            }
            return Ok(DecodedValue::TextArray(values));
        }
        if compressed && code == 3 {
            let encoded = fast_decompress_limited(
                cursor.read_bytes(cursor.remaining())?,
                integer_working_size(count)?,
            )?;
            let values = decode_integer_data(&encoded, count)?
                .into_iter()
                .map(f64::from)
                .collect();
            return Ok(DecodedValue::Numbers {
                type_code: code,
                values,
            });
        }
        let component_count = match code {
            20 => 2,
            23 | 24 | 26 => 3,
            27 => 4,
            _ => 1,
        };
        let component_size = match code {
            1 | 2 => 1,
            3 | 4 | 8 | 20 | 24 | 26 => 4,
            5 | 6 | 9 | 23 | 27 => 8,
            _ => return Err(import_error("USD crate array value type is unsupported")),
        };
        let total = count
            .checked_mul(component_count)
            .and_then(|size| size.checked_mul(component_size))
            .ok_or_else(|| import_error("USD array size overflow"))?;
        let data = if compressed && (code == 8 || code == 9) {
            let mode = cursor.read_u8()?;
            if mode == b'i' {
                let encoded = fast_decompress_limited(
                    cursor.read_bytes(cursor.remaining())?,
                    integer_working_size(count)?,
                )?;
                let ints = decode_integer_data(&encoded, count)?;
                ints.into_iter()
                    .flat_map(|value| {
                        if code == 8 {
                            (value as f32).to_le_bytes().to_vec()
                        } else {
                            f64::from(value).to_le_bytes().to_vec()
                        }
                    })
                    .collect::<Vec<_>>()
            } else if mode == b't' {
                return Err(import_error(
                    "USD floating point lookup-table arrays are unsupported",
                ));
            } else {
                return Err(import_error("unknown USD floating-point compression mode"));
            }
        } else {
            cursor.read_bytes(total)?.to_vec()
        };
        let values = if code == 3 || code == 4 || code == 26 {
            let (chunks, remainder) = data.as_chunks::<4>();
            if !remainder.is_empty() {
                return Err(import_error("truncated USD integer array"));
            }
            chunks
                .iter()
                .map(|chunk| Ok(f64::from(i32::from_le_bytes(*chunk))))
                .collect::<Result<Vec<_>>>()?
        } else if code == 1 {
            data.iter().map(|value| f64::from(*value != 0)).collect()
        } else if code == 8 || code == 20 || code == 24 {
            decode_f32_components(&data)?
        } else {
            decode_f64_components(&data)?
        };
        Ok(DecodedValue::Numbers {
            type_code: code,
            values,
        })
    } else {
        let value = if inlined { rep & 0xffff_ffff } else { 0 };
        match code {
            1 => Ok(DecodedValue::Bool(value != 0)),
            3 | 42 | 44 => Ok(DecodedValue::I32(value as u32 as i32)),
            5 => Ok(DecodedValue::I64(read_i64(bytes, payload)?)),
            8 => Ok(DecodedValue::F32(f32::from_bits(value as u32))),
            9 => Ok(DecodedValue::F64(f64::from_le_bytes(
                bytes
                    .get(payload..payload + 8)
                    .ok_or_else(|| import_error("truncated USD double"))?
                    .try_into()
                    .map_err(|_| import_error("invalid USD double"))?,
            ))),
            10 => {
                let token = *strings
                    .get(value as usize)
                    .ok_or_else(|| import_error("crate string index is out of range"))?
                    as usize;
                Ok(DecodedValue::Text(
                    tokens
                        .get(token)
                        .ok_or_else(|| import_error("crate string token is out of range"))?
                        .clone(),
                ))
            }
            11 => Ok(DecodedValue::Text(
                tokens
                    .get(value as usize)
                    .ok_or_else(|| import_error("crate token index is out of range"))?
                    .clone(),
            )),
            41 => {
                let mut vector = Cursor::at(bytes, payload)?;
                let count = usize::try_from(vector.read_u64()?)
                    .map_err(|_| import_error("USD token vector exceeds platform limits"))?;
                let mut values = Vec::with_capacity(count);
                for _ in 0..count {
                    let index = vector.read_u32()? as usize;
                    values.push(
                        tokens
                            .get(index)
                            .ok_or_else(|| import_error("USD token vector index is invalid"))?
                            .clone(),
                    );
                }
                Ok(DecodedValue::TextArray(values))
            }
            48 => {
                let mut vector = Cursor::at(bytes, payload)?;
                let count = usize::try_from(vector.read_u64()?)
                    .map_err(|_| import_error("USD double vector exceeds platform limits"))?;
                let size = count
                    .checked_mul(8)
                    .ok_or_else(|| import_error("USD double vector size overflow"))?;
                let values = decode_f64_components(vector.read_bytes(size)?)?;
                Ok(DecodedValue::Numbers {
                    type_code: 9,
                    values,
                })
            }
            15 | 16 | 20 | 23 | 24 | 26 | 27 => {
                let count = match code {
                    15 => 16,
                    20 => 2,
                    23 | 24 | 26 => 3,
                    _ => 4,
                };
                let width = if code == 20 || code == 24 || code == 26 {
                    4
                } else {
                    8
                };
                let size = count * width;
                let data = bytes
                    .get(payload..payload + size)
                    .ok_or_else(|| import_error("truncated USD vector value"))?;
                let numbers = if width == 4 {
                    decode_f32_components(data)?
                } else {
                    decode_f64_components(data)?
                };
                Ok(DecodedValue::Numbers {
                    type_code: code,
                    values: numbers,
                })
            }
            46 => decode_time_samples(payload, tokens, strings, bytes, minor),
            _ => Err(import_error("USD crate value type is unsupported")),
        }
    }
}
fn decode_f32_components(data: &[u8]) -> Result<Vec<f64>> {
    let (chunks, remainder) = data.as_chunks::<4>();
    if !remainder.is_empty() {
        return Err(import_error("truncated USD float value"));
    }
    Ok(chunks
        .iter()
        .map(|bits| f64::from(f32::from_le_bytes(*bits)))
        .collect())
}

fn decode_f64_components(data: &[u8]) -> Result<Vec<f64>> {
    let (chunks, remainder) = data.as_chunks::<8>();
    if !remainder.is_empty() {
        return Err(import_error("truncated USD double value"));
    }
    Ok(chunks
        .iter()
        .map(|bits| f64::from_le_bytes(*bits))
        .collect())
}

fn decode_time_samples(
    payload: usize,
    tokens: &[String],
    strings: &[u32],
    bytes: &[u8],
    minor: u8,
) -> Result<DecodedValue> {
    let times_relative = usize::try_from(read_i64(bytes, payload)?)
        .map_err(|_| import_error("invalid timeSamples times offset"))?;
    let times_rep_position = payload
        .checked_add(times_relative)
        .ok_or_else(|| import_error("timeSamples times offset overflow"))?;
    let times_rep = u64::from_le_bytes(
        bytes
            .get(times_rep_position..times_rep_position + 8)
            .ok_or_else(|| import_error("truncated timeSamples times rep"))?
            .try_into()
            .map_err(|_| import_error("invalid timeSamples times rep"))?,
    );
    let DecodedValue::Numbers {
        type_code: 9,
        values: times,
    } = decode_value(times_rep, tokens, strings, bytes, minor)?
    else {
        return Err(import_error(
            "USD timeSamples contains invalid sample times",
        ));
    };
    let values_pointer = times_rep_position
        .checked_add(8)
        .ok_or_else(|| import_error("timeSamples values offset overflow"))?;
    let values_relative = usize::try_from(read_i64(bytes, values_pointer)?)
        .map_err(|_| import_error("invalid timeSamples values offset"))?;
    let values_position = values_pointer
        .checked_add(values_relative)
        .ok_or_else(|| import_error("timeSamples values offset overflow"))?;
    let mut cursor = Cursor::at(bytes, values_position)?;
    let count = usize::try_from(cursor.read_u64()?)
        .map_err(|_| import_error("USD timeSamples count exceeds platform limits"))?;
    if count != times.len() {
        return Err(import_error("USD timeSamples time and value counts differ"));
    }
    let mut values = Vec::with_capacity(count);
    for _ in 0..count {
        values.push(decode_value(
            cursor.read_u64()?,
            tokens,
            strings,
            bytes,
            minor,
        )?);
    }
    Ok(DecodedValue::TimeSamples { times, values })
}

#[derive(Clone, Debug)]
struct OutPrim {
    name: String,
    kind: String,
    properties: Vec<String>,
    children: Vec<OutPrim>,
}

fn write_prim(output: &mut String, prim: &OutPrim, depth: usize) {
    let indent = "    ".repeat(depth);
    let _ = writeln!(
        output,
        "{indent}def {} \"{}\"",
        prim.kind,
        escape(&prim.name)
    );
    let _ = writeln!(output, "{indent}{{");
    for property in &prim.properties {
        let _ = writeln!(output, "{indent}    {property}");
    }
    for child in &prim.children {
        write_prim(output, child, depth + 1);
    }
    let _ = writeln!(output, "{indent}}}");
}

fn export_error(message: impl Into<String>) -> PotError {
    PotError::new(ErrorCode::ExportFailed, message)
}
fn unsupported_export(message: &str) -> PotError {
    PotError::with_details(
        ErrorCode::UnsupportedFeature,
        message,
        json!({"feature_id": "format.usdc", "status": "not_supported"}),
    )
}
fn unsupported_import(message: &str) -> PotError {
    PotError::with_details(
        ErrorCode::UnsupportedFeature,
        message,
        json!({"feature_id": "format.usdc", "status": "not_supported"}),
    )
}
fn format_number(value: f64) -> String {
    if value == 0.0 {
        "0".to_owned()
    } else {
        value.to_string()
    }
}
fn escape(value: &str) -> String {
    value
        .replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
        .replace('\r', "\\r")
}
fn align8(bytes: &mut Vec<u8>) {
    while !bytes.len().is_multiple_of(8) {
        bytes.push(0);
    }
}
fn put_u32(bytes: &mut Vec<u8>, value: u32) {
    bytes.extend_from_slice(&value.to_le_bytes());
}
fn put_i32(bytes: &mut Vec<u8>, value: i32) {
    bytes.extend_from_slice(&value.to_le_bytes());
}
fn put_u64(bytes: &mut Vec<u8>, value: u64) {
    bytes.extend_from_slice(&value.to_le_bytes());
}
fn put_i64(bytes: &mut Vec<u8>, value: i64) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

fn read_i64(bytes: &[u8], offset: usize) -> Result<i64> {
    let value = bytes
        .get(
            offset
                ..offset
                    .checked_add(8)
                    .ok_or_else(|| import_error("offset overflow"))?,
        )
        .ok_or_else(|| import_error("truncated crate integer"))?;
    Ok(i64::from_le_bytes(
        value
            .try_into()
            .map_err(|_| import_error("invalid crate integer"))?,
    ))
}

fn take_i32(bytes: &[u8], position: &mut usize) -> Result<i32> {
    let data = take_bytes(bytes, position, 4)?;
    Ok(i32::from_le_bytes(
        data.try_into()
            .map_err(|_| import_error("invalid encoded integer"))?,
    ))
}
fn take_i16(bytes: &[u8], position: &mut usize) -> Result<i16> {
    let data = take_bytes(bytes, position, 2)?;
    Ok(i16::from_le_bytes(
        data.try_into()
            .map_err(|_| import_error("invalid encoded integer"))?,
    ))
}
fn take_i8(bytes: &[u8], position: &mut usize) -> Result<i8> {
    let data = take_bytes(bytes, position, 1)?;
    Ok(data[0] as i8)
}
fn take_bytes<'a>(bytes: &'a [u8], position: &mut usize, size: usize) -> Result<&'a [u8]> {
    let end = position
        .checked_add(size)
        .ok_or_else(|| import_error("encoded integer offset overflow"))?;
    let result = bytes
        .get(*position..end)
        .ok_or_else(|| import_error("truncated encoded integer"))?;
    *position = end;
    Ok(result)
}

struct Cursor<'a> {
    bytes: &'a [u8],
    position: usize,
}
impl<'a> Cursor<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, position: 0 }
    }
    fn at(bytes: &'a [u8], position: usize) -> Result<Self> {
        if position > bytes.len() {
            return Err(import_error("crate offset is out of bounds"));
        }
        Ok(Self { bytes, position })
    }
    fn remaining(&self) -> usize {
        self.bytes.len().saturating_sub(self.position)
    }
    fn read_bytes(&mut self, size: usize) -> Result<&'a [u8]> {
        take_bytes(self.bytes, &mut self.position, size)
    }
    fn read_u8(&mut self) -> Result<u8> {
        Ok(self.read_bytes(1)?[0])
    }
    fn read_u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(super::decode_le_array(
            self.read_bytes(4)?,
            "invalid crate u32",
        )?))
    }
    fn read_u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(super::decode_le_array(
            self.read_bytes(8)?,
            "invalid crate u64",
        )?))
    }
    fn read_i32(&mut self) -> Result<i32> {
        Ok(i32::from_le_bytes(
            self.read_bytes(4)?
                .try_into()
                .map_err(|_| import_error("invalid crate i32"))?,
        ))
    }
    fn read_i64(&mut self) -> Result<i64> {
        Ok(i64::from_le_bytes(
            self.read_bytes(8)?
                .try_into()
                .map_err(|_| import_error("invalid crate i64"))?,
        ))
    }
}
