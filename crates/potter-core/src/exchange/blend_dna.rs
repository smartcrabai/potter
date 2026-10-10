use std::collections::{BTreeMap, HashSet};
use std::io::Read;

use serde_json::{Map, Number, Value};

const LEGACY_HEADER_LEN: usize = 12;
const MAX_FILE_BYTES: usize = 512 * 1024 * 1024;
const MAX_BLOCKS: usize = 250_000;
const MAX_DNA_ENTRIES: usize = 250_000;
const MAX_ARRAY_ELEMENTS: usize = 1_000_000;
const MAX_MODIFIERS_PER_OBJECT: usize = 100_000;
const MAX_BOUND_MODIFIERS: usize = 100_000;
const ZSTD_MAGIC: [u8; 4] = [0x28, 0xb5, 0x2f, 0xfd];

/// Read native bound-data fields from Blender's serialized modifier structs.
///
/// Pointer-valued members in the returned JSON retain their serialized old-address value;
/// pointers to bind payload arrays are expanded in place to native JSON arrays.
pub(crate) fn read_modifier_bindings(
    bytes: &[u8],
) -> Result<BTreeMap<(String, String), Value>, String> {
    let decompressed;
    let bytes = if bytes.starts_with(b"BLENDER") {
        bytes
    } else if bytes.starts_with(&ZSTD_MAGIC) {
        decompressed = decompress_blend(bytes)?;
        decompressed.as_slice()
    } else {
        return Err("input is neither a Blender file nor a zstd-compressed Blender file".into());
    };

    let file = BlendFile::parse(bytes)?;
    file.read_bindings()
}

fn decompress_blend(bytes: &[u8]) -> Result<Vec<u8>, String> {
    use ruzstd::decoding::errors::{FrameDecoderError, ReadFrameHeaderError};

    let max_window_size =
        u64::try_from(MAX_FILE_BYTES).map_err(|_| "maximum Blender file size is invalid")?;
    let mut remaining = bytes;
    let mut decoded = Vec::new();
    let mut frame_count = 0_usize;
    while !remaining.is_empty() {
        frame_count += 1;
        if frame_count > MAX_BLOCKS {
            return Err("Blender zstd stream contains too many frames".into());
        }
        let mut decoder = match ruzstd::decoding::StreamingDecoder::new_with_max_window_size(
            remaining,
            max_window_size,
        ) {
            Ok(decoder) => decoder,
            Err(FrameDecoderError::ReadFrameHeaderError(ReadFrameHeaderError::SkipFrame {
                length,
                ..
            })) => {
                let skip = 8_usize
                    .checked_add(usize::try_from(length).map_err(
                        |_| "Blender zstd skippable-frame length exceeds platform limits",
                    )?)
                    .ok_or_else(|| "Blender zstd skippable-frame length overflow".to_owned())?;
                remaining = remaining
                    .get(skip..)
                    .ok_or_else(|| "truncated Blender zstd skippable frame".to_owned())?;
                continue;
            }
            Err(error) => {
                return Err(format!(
                    "could not initialize Blender zstd decompressor: {error}"
                ));
            }
        };
        let before_remaining = remaining.len();
        let before_output = decoded.len();
        let output_limit = u64::try_from(
            MAX_FILE_BYTES
                .saturating_sub(decoded.len())
                .saturating_add(1),
        )
        .map_err(|_| "maximum Blender file size is invalid")?;
        decoder
            .by_ref()
            .take(output_limit)
            .read_to_end(&mut decoded)
            .map_err(|_| "could not decompress Blender zstd frame")?;
        if decoded.len() > MAX_FILE_BYTES {
            return Err("decompressed Blender file exceeds the size limit".into());
        }
        remaining = decoder.into_inner();
        if remaining.len() >= before_remaining && decoded.len() == before_output {
            return Err("Blender zstd decompressor made no progress".into());
        }
    }
    if !decoded.starts_with(b"BLENDER") {
        return Err("zstd stream does not contain a Blender file".into());
    }
    Ok(decoded)
}

#[derive(Clone, Copy, Debug)]
enum Endian {
    Little,
    Big,
}

impl Endian {
    fn read_u16(self, bytes: &[u8]) -> Result<u16, String> {
        let value: [u8; 2] = bytes
            .get(..2)
            .ok_or_else(|| "truncated 16-bit Blender value".to_owned())?
            .try_into()
            .map_err(|_| "invalid 16-bit Blender value".to_owned())?;
        Ok(match self {
            Self::Little => u16::from_le_bytes(value),
            Self::Big => u16::from_be_bytes(value),
        })
    }

    fn read_u32(self, bytes: &[u8]) -> Result<u32, String> {
        let value: [u8; 4] = bytes
            .get(..4)
            .ok_or_else(|| "truncated 32-bit Blender value".to_owned())?
            .try_into()
            .map_err(|_| "invalid 32-bit Blender value".to_owned())?;
        Ok(match self {
            Self::Little => u32::from_le_bytes(value),
            Self::Big => u32::from_be_bytes(value),
        })
    }

    fn read_u64(self, bytes: &[u8]) -> Result<u64, String> {
        let value: [u8; 8] = bytes
            .get(..8)
            .ok_or_else(|| "truncated 64-bit Blender value".to_owned())?
            .try_into()
            .map_err(|_| "invalid 64-bit Blender value".to_owned())?;
        Ok(match self {
            Self::Little => u64::from_le_bytes(value),
            Self::Big => u64::from_be_bytes(value),
        })
    }
}

#[derive(Clone, Copy, Debug)]
struct Header {
    pointer_size: usize,
    endian: Endian,
    preamble_len: usize,
    large_bhead: bool,
}

impl Header {
    fn parse(bytes: &[u8]) -> Result<Self, String> {
        if bytes.len() < LEGACY_HEADER_LEN || !bytes.starts_with(b"BLENDER") {
            return Err("invalid or truncated Blender file header".into());
        }
        if bytes[7..9].iter().all(u8::is_ascii_digit) {
            if bytes.len() < 17
                || bytes[9] != b'-'
                || &bytes[10..12] != b"01"
                || bytes[12] != b'v'
                || !bytes[13..17].iter().all(u8::is_ascii_digit)
            {
                return Err("invalid Blender large-block header".into());
            }
            let header_size = std::str::from_utf8(&bytes[7..9])
                .ok()
                .and_then(|value| value.parse::<usize>().ok())
                .filter(|value| *value == 17)
                .ok_or_else(|| "unsupported Blender large-block header size".to_owned())?;
            return Ok(Self {
                pointer_size: 8,
                endian: Endian::Little,
                preamble_len: header_size,
                large_bhead: true,
            });
        }
        let pointer_size = match bytes[7] {
            b'_' => 4,
            b'-' => 8,
            _ => return Err("invalid Blender pointer-size marker".into()),
        };
        let endian = match bytes[8] {
            b'v' => Endian::Little,
            b'V' => Endian::Big,
            _ => return Err("invalid Blender byte-order marker".into()),
        };
        if !bytes[9..12].iter().all(u8::is_ascii_digit) {
            return Err("invalid Blender version marker".into());
        }
        Ok(Self {
            pointer_size,
            endian,
            preamble_len: LEGACY_HEADER_LEN,
            large_bhead: false,
        })
    }

    fn read_pointer(self, bytes: &[u8]) -> Result<u64, String> {
        match self.pointer_size {
            4 => Ok(u64::from(self.endian.read_u32(bytes)?)),
            8 => self.endian.read_u64(bytes),
            _ => Err("unsupported Blender pointer size".into()),
        }
    }
}

#[derive(Clone, Debug)]
struct Block {
    code: [u8; 4],
    len: usize,
    old_address: u64,
    sdna_index: usize,
    count: usize,
    data_offset: usize,
}

impl Block {
    fn data_range(&self) -> Result<std::ops::Range<usize>, String> {
        let end = self
            .data_offset
            .checked_add(self.len)
            .ok_or_else(|| "Blender block range overflow".to_owned())?;
        Ok(self.data_offset..end)
    }
}

#[derive(Clone, Debug)]
struct RawField {
    type_index: usize,
    name_index: usize,
}

#[derive(Clone, Debug)]
struct RawStruct {
    type_index: usize,
    fields: Vec<RawField>,
}

#[derive(Clone, Debug)]
struct Field {
    type_index: usize,
    name: String,
    pointer: bool,
    dimensions: Vec<usize>,
    offset: usize,
    size: usize,
}

#[derive(Clone, Debug)]
struct StructDef {
    type_index: usize,
    fields: Vec<Field>,
    size: usize,
}

#[derive(Clone, Debug)]
struct Schema {
    type_names: Vec<String>,
    type_lengths: Vec<usize>,
    structs: Vec<StructDef>,
    struct_by_type: Vec<Option<usize>>,
}

impl Schema {
    fn parse(bytes: &[u8], endian: Endian, pointer_size: usize) -> Result<Self, String> {
        let mut cursor = DnaCursor::new(bytes, endian);
        cursor.expect(b"SDNA")?;
        cursor.expect(b"NAME")?;
        let name_count = cursor.read_count("DNA names")?;
        let mut names = Vec::with_capacity(name_count);
        for _ in 0..name_count {
            names.push(cursor.read_c_string()?);
        }
        cursor.align4()?;

        cursor.expect(b"TYPE")?;
        let type_count = cursor.read_count("DNA types")?;
        let mut type_names = Vec::with_capacity(type_count);
        for _ in 0..type_count {
            type_names.push(cursor.read_c_string()?);
        }
        cursor.align4()?;

        cursor.expect(b"TLEN")?;
        let mut type_lengths = Vec::with_capacity(type_count);
        for _ in 0..type_count {
            type_lengths.push(usize::from(cursor.read_u16()?));
        }
        cursor.align4()?;

        cursor.expect(b"STRC")?;
        let struct_count = cursor.read_count("DNA structures")?;
        let mut raw_structs = Vec::with_capacity(struct_count);
        let mut struct_by_type = vec![None; type_count];
        for struct_index in 0..struct_count {
            let type_index = usize::from(cursor.read_u16()?);
            let field_count = usize::from(cursor.read_u16()?);
            if type_index >= type_count || field_count > MAX_DNA_ENTRIES {
                return Err("invalid Blender DNA structure declaration".into());
            }
            if struct_by_type[type_index].replace(struct_index).is_some() {
                return Err("duplicate Blender DNA structure type".into());
            }
            let mut fields = Vec::with_capacity(field_count);
            for _ in 0..field_count {
                let field_type = usize::from(cursor.read_u16()?);
                let name_index = usize::from(cursor.read_u16()?);
                if field_type >= type_count || name_index >= names.len() {
                    return Err("Blender DNA field index is out of range".into());
                }
                fields.push(RawField {
                    type_index: field_type,
                    name_index,
                });
            }
            raw_structs.push(RawStruct { type_index, fields });
        }
        if cursor.position != bytes.len() {
            return Err("unexpected trailing bytes in Blender DNA schema".into());
        }

        let mut structs: Vec<Option<StructDef>> = vec![None; raw_structs.len()];
        for index in 0..raw_structs.len() {
            build_struct_layout(
                index,
                pointer_size,
                &raw_structs,
                &names,
                &type_names,
                &type_lengths,
                &mut structs,
            )?;
        }
        let structs = structs
            .into_iter()
            .map(|entry| entry.ok_or_else(|| "incomplete Blender DNA schema".to_owned()))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            type_names,
            type_lengths,
            structs,
            struct_by_type,
        })
    }

    fn struct_index(&self, type_name: &str) -> Option<usize> {
        self.type_names
            .iter()
            .position(|name| name == type_name)
            .and_then(|type_index| self.struct_by_type.get(type_index).copied().flatten())
    }

    fn struct_for_type(&self, type_index: usize) -> Option<usize> {
        self.struct_by_type.get(type_index).copied().flatten()
    }

    fn field(&self, struct_index: usize, name: &str) -> Option<&Field> {
        self.structs
            .get(struct_index)?
            .fields
            .iter()
            .find(|field| field.name == name)
    }

    fn field_address(
        &self,
        struct_index: usize,
        base_offset: usize,
        path: &[&str],
    ) -> Result<(usize, &Field), String> {
        let (last, parents) = path
            .split_last()
            .ok_or_else(|| "empty Blender DNA field path".to_owned())?;
        let mut current_struct = struct_index;
        let mut address = base_offset;
        for parent in parents {
            let field = self
                .field(current_struct, parent)
                .ok_or_else(|| format!("Blender DNA field path {} is missing", path.join(".")))?;
            if field.pointer || !field.dimensions.is_empty() {
                return Err(format!(
                    "Blender DNA field path {} crosses a non-inline member",
                    path.join(".")
                ));
            }
            address = address
                .checked_add(field.offset)
                .ok_or_else(|| "Blender DNA field offset overflow".to_owned())?;
            current_struct = self.struct_for_type(field.type_index).ok_or_else(|| {
                format!(
                    "Blender DNA field path {} is not a structure",
                    path.join(".")
                )
            })?;
        }
        let field = self
            .field(current_struct, last)
            .ok_or_else(|| format!("Blender DNA field path {} is missing", path.join(".")))?;
        address = address
            .checked_add(field.offset)
            .ok_or_else(|| "Blender DNA field offset overflow".to_owned())?;
        Ok((address, field))
    }
    fn decode_struct(
        &self,
        bytes: &[u8],
        struct_index: usize,
        base_offset: usize,
        endian: Endian,
        pointer_size: usize,
        depth: usize,
    ) -> Result<Value, String> {
        if depth > 64 {
            return Err("Blender DNA nesting exceeds the depth limit".into());
        }
        let definition = self
            .structs
            .get(struct_index)
            .ok_or_else(|| "Blender DNA structure index is invalid".to_owned())?;
        checked_slice(bytes, base_offset, definition.size)?;
        let mut object = Map::new();
        for field in &definition.fields {
            let address = base_offset
                .checked_add(field.offset)
                .ok_or_else(|| "Blender DNA field offset overflow".to_owned())?;
            let value =
                self.decode_field(bytes, field, address, endian, pointer_size, depth + 1)?;
            object.insert(field.name.clone(), value);
        }
        Ok(Value::Object(object))
    }

    fn decode_field(
        &self,
        bytes: &[u8],
        field: &Field,
        address: usize,
        endian: Endian,
        pointer_size: usize,
        depth: usize,
    ) -> Result<Value, String> {
        if field.pointer {
            if field.dimensions.is_empty() {
                let pointer = read_pointer_at(bytes, address, pointer_size, endian)?;
                return Ok(Value::Number(Number::from(pointer)));
            }
            return decode_pointer_array(
                bytes,
                &field.dimensions,
                address,
                pointer_size,
                endian,
                depth + 1,
            );
        }
        if !field.dimensions.is_empty() {
            if self.type_names[field.type_index] == "char"
                && field.dimensions.len() == 1
                && !field.name.starts_with("_pad")
                && !field.name.starts_with("pad")
            {
                let raw = checked_slice(bytes, address, field.size)?;
                return Ok(Value::String(decode_c_string(raw)));
            }
            return self.decode_array(
                bytes,
                field.type_index,
                &field.dimensions,
                address,
                endian,
                pointer_size,
                depth + 1,
            );
        }
        self.decode_type(
            bytes,
            field.type_index,
            address,
            endian,
            pointer_size,
            depth + 1,
        )
    }

    fn decode_array(
        &self,
        bytes: &[u8],
        type_index: usize,
        dimensions: &[usize],
        address: usize,
        endian: Endian,
        pointer_size: usize,
        depth: usize,
    ) -> Result<Value, String> {
        if depth > 64 || dimensions.is_empty() {
            return Err("invalid or excessively nested Blender DNA array".into());
        }
        let count = dimensions[0];
        if count > MAX_ARRAY_ELEMENTS {
            return Err("Blender DNA array exceeds the element limit".into());
        }
        let element_size = if dimensions.len() == 1 {
            self.type_lengths[type_index]
        } else {
            self.type_lengths[type_index]
                .checked_mul(
                    dimensions[1..]
                        .iter()
                        .try_fold(1_usize, |product, value| product.checked_mul(*value))
                        .ok_or_else(|| "Blender DNA array dimensions overflow".to_owned())?,
                )
                .ok_or_else(|| "Blender DNA array byte size overflow".to_owned())?
        };
        let mut values = Vec::with_capacity(count);
        for index in 0..count {
            let item_offset = address
                .checked_add(
                    index
                        .checked_mul(element_size)
                        .ok_or_else(|| "Blender DNA array element offset overflow".to_owned())?,
                )
                .ok_or_else(|| "Blender DNA array offset overflow".to_owned())?;
            let value = if dimensions.len() == 1 {
                self.decode_type(
                    bytes,
                    type_index,
                    item_offset,
                    endian,
                    pointer_size,
                    depth + 1,
                )?
            } else {
                self.decode_array(
                    bytes,
                    type_index,
                    &dimensions[1..],
                    item_offset,
                    endian,
                    pointer_size,
                    depth + 1,
                )?
            };
            values.push(value);
        }
        Ok(Value::Array(values))
    }

    fn decode_type(
        &self,
        bytes: &[u8],
        type_index: usize,
        address: usize,
        endian: Endian,
        pointer_size: usize,
        depth: usize,
    ) -> Result<Value, String> {
        if let Some(struct_index) = self.struct_for_type(type_index) {
            return self.decode_struct(
                bytes,
                struct_index,
                address,
                endian,
                pointer_size,
                depth + 1,
            );
        }
        let type_name = &self.type_names[type_index];
        let size = self.type_lengths[type_index];
        let raw = checked_slice(bytes, address, size)?;
        if type_name == "float" {
            if size != 4 {
                return Err("unsupported Blender DNA float width".into());
            }
            let bits = endian.read_u32(raw)?;
            return float_value(f64::from(f32::from_bits(bits)));
        }
        if type_name == "double" {
            if size != 8 {
                return Err("unsupported Blender DNA double width".into());
            }
            let bits = endian.read_u64(raw)?;
            return float_value(f64::from_bits(bits));
        }
        match size {
            1 => {
                let value = raw[0];
                if is_unsigned_type(type_name) {
                    Ok(Value::Number(Number::from(u64::from(value))))
                } else {
                    Ok(Value::Number(Number::from(i64::from(value as i8))))
                }
            }
            2 => {
                let value = endian.read_u16(raw)?;
                if is_unsigned_type(type_name) {
                    Ok(Value::Number(Number::from(u64::from(value))))
                } else {
                    Ok(Value::Number(Number::from(i64::from(value as i16))))
                }
            }
            4 => {
                let value = endian.read_u32(raw)?;
                if is_unsigned_type(type_name) {
                    Ok(Value::Number(Number::from(u64::from(value))))
                } else {
                    Ok(Value::Number(Number::from(i64::from(value as i32))))
                }
            }
            8 => {
                let value = endian.read_u64(raw)?;
                if is_unsigned_type(type_name) {
                    Ok(Value::Number(Number::from(value)))
                } else {
                    Ok(Value::Number(Number::from(value as i64)))
                }
            }
            _ => Err(format!("unsupported Blender DNA scalar type {type_name}")),
        }
    }

    fn pointer_at_path(
        &self,
        bytes: &[u8],
        struct_index: usize,
        base_offset: usize,
        path: &[&str],
        endian: Endian,
        pointer_size: usize,
    ) -> Result<u64, String> {
        let (address, field) = self.field_address(struct_index, base_offset, path)?;
        if !field.pointer {
            return Err(format!(
                "Blender DNA field {} is not a pointer",
                path.join(".")
            ));
        }
        read_pointer_at(bytes, address, pointer_size, endian)
    }
    fn optional_pointer_at_path(
        &self,
        bytes: &[u8],
        struct_index: usize,
        base_offset: usize,
        path: &[&str],
        endian: Endian,
        pointer_size: usize,
    ) -> Result<u64, String> {
        let Ok((address, field)) = self.field_address(struct_index, base_offset, path) else {
            return Ok(0);
        };
        if !field.pointer {
            return Err(format!(
                "Blender DNA field {} is not a pointer",
                path.join(".")
            ));
        }
        read_pointer_at(bytes, address, pointer_size, endian)
    }

    fn integer_at_path(
        &self,
        bytes: &[u8],
        struct_index: usize,
        base_offset: usize,
        path: &[&str],
        endian: Endian,
    ) -> Result<Option<i64>, String> {
        let Ok((address, field)) = self.field_address(struct_index, base_offset, path) else {
            return Ok(None);
        };
        if field.pointer || !field.dimensions.is_empty() {
            return Err(format!(
                "Blender DNA field {} is not an integer",
                path.join(".")
            ));
        }
        let value = self.decode_type(bytes, field.type_index, address, endian, 8, 0)?;
        value
            .as_i64()
            .map(Some)
            .ok_or_else(|| format!("Blender DNA field {} is not an integer", path.join(".")))
    }

    fn string_at_path(
        &self,
        bytes: &[u8],
        struct_index: usize,
        base_offset: usize,
        path: &[&str],
    ) -> Result<String, String> {
        let (address, field) = self.field_address(struct_index, base_offset, path)?;
        if field.pointer {
            return Err(format!(
                "Blender DNA field {} is not inline text",
                path.join(".")
            ));
        }
        if !field.dimensions.is_empty() && self.type_names[field.type_index] == "char" {
            return Ok(decode_c_string(checked_slice(bytes, address, field.size)?));
        }
        let value = self.decode_type(bytes, field.type_index, address, Endian::Little, 8, 0)?;
        value
            .as_str()
            .map(str::to_owned)
            .ok_or_else(|| format!("Blender DNA field {} is not text", path.join(".")))
    }
}

// Blender SDNA stores explicit padding members, so serialized member offsets are packed sums.
fn build_struct_layout(
    struct_index: usize,
    pointer_size: usize,
    raw_structs: &[RawStruct],
    names: &[String],
    type_names: &[String],
    type_lengths: &[usize],
    output: &mut [Option<StructDef>],
) -> Result<(), String> {
    let raw = &raw_structs[struct_index];
    let mut fields = Vec::with_capacity(raw.fields.len());
    let mut offset = 0_usize;
    for raw_field in &raw.fields {
        let raw_name = names
            .get(raw_field.name_index)
            .ok_or_else(|| "Blender DNA member name index is invalid".to_owned())?;
        let declaration = parse_declarator(raw_name)?;
        let element_size = if declaration.pointer {
            pointer_size
        } else {
            *type_lengths
                .get(raw_field.type_index)
                .ok_or_else(|| "Blender DNA type index is invalid".to_owned())?
        };
        if element_size == 0 {
            return Err("zero-sized Blender DNA member type".into());
        }
        let count = declaration
            .dimensions
            .iter()
            .try_fold(1_usize, |product, value| product.checked_mul(*value))
            .ok_or_else(|| "Blender DNA array dimensions overflow".to_owned())?;
        if count > MAX_ARRAY_ELEMENTS {
            return Err("Blender DNA member array exceeds the element limit".into());
        }
        let member_size = element_size
            .checked_mul(count)
            .ok_or_else(|| "Blender DNA member size overflow".to_owned())?;
        fields.push(Field {
            type_index: raw_field.type_index,
            name: declaration.name,
            pointer: declaration.pointer,
            dimensions: declaration.dimensions,
            offset,
            size: member_size,
        });
        offset = offset
            .checked_add(member_size)
            .ok_or_else(|| "Blender DNA structure size overflow".to_owned())?;
    }
    let expected_size = *type_lengths
        .get(raw.type_index)
        .ok_or_else(|| "Blender DNA structure type is invalid".to_owned())?;
    if offset != expected_size {
        return Err(format!(
            "Blender DNA layout mismatch for {} (computed {offset}, declared {expected_size})",
            type_names[raw.type_index]
        ));
    }
    output[struct_index] = Some(StructDef {
        type_index: raw.type_index,
        fields,
        size: offset,
    });
    Ok(())
}
struct Declarator {
    name: String,
    pointer: bool,
    dimensions: Vec<usize>,
}

fn parse_declarator(declaration: &str) -> Result<Declarator, String> {
    let bytes = declaration.as_bytes();
    let function_pointer = declaration.find("(*");
    let (start, pointer) = if let Some(group_start) = function_pointer {
        (group_start + 2, true)
    } else {
        let mut start = 0;
        while bytes
            .get(start)
            .is_some_and(|byte| byte.is_ascii_whitespace() || *byte == b'*')
        {
            start += 1;
        }
        (start, declaration[..start].contains('*'))
    };
    let mut end = start;
    while bytes
        .get(end)
        .is_some_and(|byte| *byte == b'_' || byte.is_ascii_alphanumeric())
    {
        end += 1;
    }
    if end == start {
        return Err("invalid Blender DNA member declaration".into());
    }
    let name = declaration[start..end].to_owned();
    let mut dimensions = Vec::new();
    let mut cursor = end;
    while bytes.get(cursor) == Some(&b'[') {
        let close = bytes[cursor + 1..]
            .iter()
            .position(|byte| *byte == b']')
            .map(|position| cursor + 1 + position)
            .ok_or_else(|| "unterminated Blender DNA array dimension".to_owned())?;
        let dimension = declaration[cursor + 1..close]
            .parse::<usize>()
            .map_err(|_| "invalid Blender DNA array dimension".to_owned())?;
        if dimension == 0 {
            return Err("zero-sized Blender DNA array dimension".into());
        }
        dimensions.push(dimension);
        cursor = close + 1;
    }
    if function_pointer.is_none() && declaration[cursor..].contains('[') {
        return Err("unsupported Blender DNA member declarator".into());
    }
    Ok(Declarator {
        name,
        pointer,
        dimensions,
    })
}

fn decode_pointer_array(
    bytes: &[u8],
    dimensions: &[usize],
    address: usize,
    pointer_size: usize,
    endian: Endian,
    depth: usize,
) -> Result<Value, String> {
    if dimensions.is_empty() || depth > 64 {
        return Err("invalid or excessively nested Blender DNA pointer array".into());
    }
    let count = dimensions[0];
    if count > MAX_ARRAY_ELEMENTS {
        return Err("Blender DNA pointer array exceeds the element limit".into());
    }
    let item_size = if dimensions.len() == 1 {
        pointer_size
    } else {
        pointer_size
            .checked_mul(
                dimensions[1..]
                    .iter()
                    .try_fold(1_usize, |product, value| product.checked_mul(*value))
                    .ok_or_else(|| "Blender DNA pointer array dimensions overflow".to_owned())?,
            )
            .ok_or_else(|| "Blender DNA pointer array size overflow".to_owned())?
    };
    let mut values = Vec::with_capacity(count);
    for index in 0..count {
        let item_offset = address
            .checked_add(
                index
                    .checked_mul(item_size)
                    .ok_or_else(|| "Blender DNA pointer array offset overflow".to_owned())?,
            )
            .ok_or_else(|| "Blender DNA pointer array offset overflow".to_owned())?;
        let value = if dimensions.len() == 1 {
            Value::Number(Number::from(read_pointer_at(
                bytes,
                item_offset,
                pointer_size,
                endian,
            )?))
        } else {
            decode_pointer_array(
                bytes,
                &dimensions[1..],
                item_offset,
                pointer_size,
                endian,
                depth + 1,
            )?
        };
        values.push(value);
    }
    Ok(Value::Array(values))
}

fn align_up(value: usize, alignment: usize) -> Result<usize, String> {
    let remainder = value % alignment;
    if remainder == 0 {
        Ok(value)
    } else {
        value
            .checked_add(alignment - remainder)
            .ok_or_else(|| "Blender DNA alignment overflow".to_owned())
    }
}

fn is_unsigned_type(type_name: &str) -> bool {
    type_name.starts_with("unsigned ")
        || type_name.starts_with("uint")
        || type_name.starts_with("uchar")
        || type_name.starts_with("ushort")
}

fn float_value(value: f64) -> Result<Value, String> {
    Number::from_f64(value)
        .map(Value::Number)
        .ok_or_else(|| "non-finite floating-point value in Blender bind data".into())
}

fn decode_c_string(bytes: &[u8]) -> String {
    let end = bytes
        .iter()
        .position(|byte| *byte == 0)
        .unwrap_or(bytes.len());
    String::from_utf8_lossy(&bytes[..end]).into_owned()
}

fn checked_slice(bytes: &[u8], offset: usize, length: usize) -> Result<&[u8], String> {
    let end = offset
        .checked_add(length)
        .ok_or_else(|| "Blender data range overflow".to_owned())?;
    bytes
        .get(offset..end)
        .ok_or_else(|| "Blender data range is truncated".to_owned())
}

fn read_pointer_at(
    bytes: &[u8],
    address: usize,
    pointer_size: usize,
    endian: Endian,
) -> Result<u64, String> {
    let raw = checked_slice(bytes, address, pointer_size)?;
    match pointer_size {
        4 => Ok(u64::from(endian.read_u32(raw)?)),
        8 => endian.read_u64(raw),
        _ => Err("unsupported Blender pointer size".into()),
    }
}

struct DnaCursor<'a> {
    bytes: &'a [u8],
    endian: Endian,
    position: usize,
}

impl<'a> DnaCursor<'a> {
    fn new(bytes: &'a [u8], endian: Endian) -> Self {
        Self {
            bytes,
            endian,
            position: 0,
        }
    }

    fn expect(&mut self, expected: &[u8]) -> Result<(), String> {
        let actual = checked_slice(self.bytes, self.position, expected.len())?;
        if actual != expected {
            return Err("invalid Blender DNA section marker".into());
        }
        self.position += expected.len();
        Ok(())
    }

    fn read_u16(&mut self) -> Result<u16, String> {
        let value = self
            .endian
            .read_u16(checked_slice(self.bytes, self.position, 2)?)?;
        self.position += 2;
        Ok(value)
    }

    fn read_count(&mut self, label: &str) -> Result<usize, String> {
        let count = usize::try_from(self.endian.read_u32(checked_slice(
            self.bytes,
            self.position,
            4,
        )?)?)
        .map_err(|_| format!("{label} count exceeds platform limits"))?;
        self.position += 4;
        if count > MAX_DNA_ENTRIES {
            return Err(format!("{label} count exceeds the limit"));
        }
        Ok(count)
    }

    fn read_c_string(&mut self) -> Result<String, String> {
        let remaining = self
            .bytes
            .get(self.position..)
            .ok_or_else(|| "invalid Blender DNA string offset".to_owned())?;
        let length = remaining
            .iter()
            .position(|byte| *byte == 0)
            .ok_or_else(|| "unterminated Blender DNA string".to_owned())?;
        let text = String::from_utf8_lossy(&remaining[..length]).into_owned();
        self.position = self
            .position
            .checked_add(length + 1)
            .ok_or_else(|| "Blender DNA string offset overflow".to_owned())?;
        Ok(text)
    }

    fn align4(&mut self) -> Result<(), String> {
        self.position = align_up(self.position, 4)?;
        if self.position > self.bytes.len() {
            return Err("truncated Blender DNA padding".into());
        }
        Ok(())
    }
}

struct BlendFile<'a> {
    bytes: &'a [u8],
    header: Header,
    blocks: Vec<Block>,
    by_address: BTreeMap<u64, usize>,
    schema: Schema,
}

impl<'a> BlendFile<'a> {
    fn parse(bytes: &'a [u8]) -> Result<Self, String> {
        if bytes.len() > MAX_FILE_BYTES {
            return Err("Blender file exceeds the size limit".into());
        }
        let header = Header::parse(bytes)?;
        let blocks = parse_blocks(bytes, header)?;
        let dna_blocks: Vec<&Block> = blocks
            .iter()
            .filter(|block| &block.code == b"DNA1")
            .collect();
        if dna_blocks.len() != 1 {
            return Err("Blender file must contain exactly one DNA1 block".into());
        }
        let dna_range = dna_blocks[0].data_range()?;
        let schema = Schema::parse(
            checked_slice(bytes, dna_range.start, dna_range.len())?,
            header.endian,
            header.pointer_size,
        )?;
        let mut by_address = BTreeMap::new();
        let mut ambiguous_addresses = HashSet::new();
        for (index, block) in blocks.iter().enumerate() {
            let synthetic_address = block.old_address == 0x4154_4144;
            let non_pointer_block = matches!(&block.code, b"REND" | b"GLOB" | b"TEST");
            if block.old_address == 0
                || block.len == 0
                || synthetic_address
                || non_pointer_block
                || ambiguous_addresses.contains(&block.old_address)
            {
                continue;
            }
            if by_address.insert(block.old_address, index).is_some() {
                by_address.remove(&block.old_address);
                ambiguous_addresses.insert(block.old_address);
            }
        }
        Ok(Self {
            bytes,
            header,
            blocks,
            by_address,
            schema,
        })
    }

    fn read_bindings(&self) -> Result<BTreeMap<(String, String), Value>, String> {
        let Some(object_struct) = self.schema.struct_index("Object") else {
            return Err("Blender DNA schema does not contain Object".into());
        };
        let modifier_structs = [
            ("SurfaceDeformModifierData", ModifierKind::Surface),
            ("MeshDeformModifierData", ModifierKind::Mesh),
            ("LaplacianDeformModifierData", ModifierKind::Laplacian),
        ];
        let mut output = BTreeMap::new();
        for block in &self.blocks {
            if block.sdna_index >= self.schema.structs.len()
                || self.schema.structs[block.sdna_index].type_index
                    != self.schema.structs[object_struct].type_index
            {
                continue;
            }
            if block.count > MAX_ARRAY_ELEMENTS {
                return Err("Blender Object block exceeds the element limit".into());
            }
            let object_size = self.schema.structs[object_struct].size;
            if object_size == 0 {
                return Err("Blender Object SDNA structure has zero size".into());
            }
            let total_size = object_size
                .checked_mul(block.count)
                .ok_or_else(|| "Blender Object block size overflow".to_owned())?;
            if total_size > block.len {
                return Err("Blender Object block is shorter than its SDNA count".into());
            }
            for object_index in 0..block.count {
                let object_offset = block
                    .data_offset
                    .checked_add(
                        object_index
                            .checked_mul(object_size)
                            .ok_or_else(|| "Blender Object offset overflow".to_owned())?,
                    )
                    .ok_or_else(|| "Blender Object offset overflow".to_owned())?;
                let full_id_name = self.schema.string_at_path(
                    self.bytes,
                    object_struct,
                    object_offset,
                    &["id", "name"],
                )?;
                let object_name = full_id_name.get(2..).unwrap_or(&full_id_name).to_owned();
                let first_modifier = self.schema.pointer_at_path(
                    self.bytes,
                    object_struct,
                    object_offset,
                    &["modifiers", "first"],
                    self.header.endian,
                    self.header.pointer_size,
                )?;
                let mut modifier_address = first_modifier;
                let mut visited = HashSet::new();
                let mut modifier_count = 0_usize;
                while modifier_address != 0 {
                    modifier_count += 1;
                    if modifier_count > MAX_MODIFIERS_PER_OBJECT {
                        return Err("Blender modifier list exceeds the per-object limit".into());
                    }
                    if !visited.insert(modifier_address) {
                        return Err("cycle in Blender Object modifier list".into());
                    }
                    let (candidate_block, _) = self.resolve(modifier_address, 1)?;
                    let candidate_struct = candidate_block.sdna_index;
                    if candidate_struct >= self.schema.structs.len() {
                        return Err("Blender modifier block has an invalid SDNA index".into());
                    }
                    let modifier_size = self.schema.structs[candidate_struct].size;
                    let (modifier_block, modifier_offset) =
                        self.resolve(modifier_address, modifier_size)?;
                    let modifier_struct = modifier_block.sdna_index;
                    if modifier_struct != candidate_struct
                        || modifier_size == 0
                        || (modifier_offset - modifier_block.data_offset) % modifier_size != 0
                    {
                        return Err("Blender modifier pointer is not structure-aligned".into());
                    }
                    let modifier_name = self
                        .schema
                        .string_at_path(
                            self.bytes,
                            modifier_struct,
                            modifier_offset,
                            &["modifier", "name"],
                        )
                        .or_else(|_| {
                            self.schema.string_at_path(
                                self.bytes,
                                modifier_struct,
                                modifier_offset,
                                &["name"],
                            )
                        })?;
                    let type_name = self.schema.type_names
                        [self.schema.structs[modifier_struct].type_index]
                        .as_str();
                    if let Some((_, kind)) = modifier_structs
                        .iter()
                        .find(|(candidate, _)| *candidate == type_name)
                        && let Some(value) =
                            self.read_modifier_payload(modifier_struct, modifier_offset, *kind)?
                    {
                        let key = (object_name.clone(), modifier_name);
                        if output.contains_key(&key) {
                            return Err("duplicate bound Blender modifier key".into());
                        }
                        if output.len() >= MAX_BOUND_MODIFIERS {
                            return Err("Blender bind result exceeds the entry limit".into());
                        }
                        output.insert(key, value);
                    }
                    modifier_address = self
                        .schema
                        .pointer_at_path(
                            self.bytes,
                            modifier_struct,
                            modifier_offset,
                            &["modifier", "next"],
                            self.header.endian,
                            self.header.pointer_size,
                        )
                        .or_else(|_| {
                            self.schema.pointer_at_path(
                                self.bytes,
                                modifier_struct,
                                modifier_offset,
                                &["next"],
                                self.header.endian,
                                self.header.pointer_size,
                            )
                        })?;
                }
            }
        }
        Ok(output)
    }

    fn read_modifier_payload(
        &self,
        struct_index: usize,
        base_offset: usize,
        kind: ModifierKind,
    ) -> Result<Option<Value>, String> {
        let flag_name = match kind {
            ModifierKind::Surface => "flags",
            ModifierKind::Mesh | ModifierKind::Laplacian => "flag",
        };
        let flags = self
            .schema
            .integer_at_path(
                self.bytes,
                struct_index,
                base_offset,
                &[flag_name],
                self.header.endian,
            )?
            .unwrap_or(0) as u64;
        match kind {
            ModifierKind::Surface => self.read_surface_payload(struct_index, base_offset, flags),
            ModifierKind::Mesh => self.read_mesh_payload(struct_index, base_offset, flags),
            ModifierKind::Laplacian => {
                self.read_laplacian_payload(struct_index, base_offset, flags)
            }
        }
    }

    fn read_surface_payload(
        &self,
        struct_index: usize,
        base_offset: usize,
        flags: u64,
    ) -> Result<Option<Value>, String> {
        const SURFACE_BIND: u64 = 1;
        if flags & SURFACE_BIND == 0 {
            return Ok(None);
        }
        let count = self
            .schema
            .integer_at_path(
                self.bytes,
                struct_index,
                base_offset,
                &["bind_verts_num"],
                self.header.endian,
            )?
            .or(self.schema.integer_at_path(
                self.bytes,
                struct_index,
                base_offset,
                &["numverts"],
                self.header.endian,
            )?)
            .or(self.schema.integer_at_path(
                self.bytes,
                struct_index,
                base_offset,
                &["mesh_verts_num"],
                self.header.endian,
            )?)
            .unwrap_or(0);
        let count = checked_count(count, "Surface Deform bound vertex")?;
        let source_count = checked_count(
            self.schema
                .integer_at_path(
                    self.bytes,
                    struct_index,
                    base_offset,
                    &["mesh_verts_num"],
                    self.header.endian,
                )?
                .or(self.schema.integer_at_path(
                    self.bytes,
                    struct_index,
                    base_offset,
                    &["num_mesh_verts"],
                    self.header.endian,
                )?)
                .unwrap_or(i64::try_from(count).unwrap_or(i64::MAX)),
            "Surface Deform source vertex",
        )?;
        let target_count = checked_count(
            self.schema
                .integer_at_path(
                    self.bytes,
                    struct_index,
                    base_offset,
                    &["target_verts_num"],
                    self.header.endian,
                )?
                .unwrap_or(0),
            "Surface Deform target vertex",
        )?;
        let pointer = self.schema.pointer_at_path(
            self.bytes,
            struct_index,
            base_offset,
            &["verts"],
            self.header.endian,
            self.header.pointer_size,
        )?;
        if count == 0 {
            return Ok(None);
        }
        if pointer == 0 {
            return Err("bound Surface Deform has vertices but no bind array".into());
        }
        let verts_type = self
            .schema
            .type_names
            .iter()
            .position(|name| name == "SDefVert")
            .and_then(|type_index| self.schema.struct_for_type(type_index))
            .ok_or_else(|| "Blender DNA schema does not contain SDefVert".to_owned())?;
        let bind_type = self
            .schema
            .type_names
            .iter()
            .position(|name| name == "SDefBind")
            .and_then(|type_index| self.schema.struct_for_type(type_index))
            .ok_or_else(|| "Blender DNA schema does not contain SDefBind".to_owned())?;
        let verts_size = self.schema.structs[verts_type].size;
        let verts_bytes = verts_size
            .checked_mul(count)
            .ok_or_else(|| "Surface Deform vertex array size overflow".to_owned())?;
        let (_, verts_base) = self.resolve(pointer, verts_bytes)?;
        let vertex_records = self.read_struct_array(pointer, count, verts_type)?;
        let mut verts = Vec::with_capacity(count);
        let mut payload_found = false;
        for (vertex_offset, mut vertex) in vertex_records.into_iter().enumerate() {
            let vertex_address = checked_array_item_offset(verts_base, vertex_offset, verts_size)?;
            let vertex_index = vertex
                .get("vertex_idx")
                .and_then(json_nonnegative_index)
                .ok_or_else(|| "Surface Deform vertex has an invalid vertex index".to_owned())?;
            if source_count > 0 && vertex_index >= source_count {
                return Err("Surface Deform vertex index exceeds the source vertex count".into());
            }
            let binds_num = self
                .schema
                .integer_at_path(
                    self.bytes,
                    verts_type,
                    vertex_address,
                    &["binds_num"],
                    self.header.endian,
                )?
                .or(self.schema.integer_at_path(
                    self.bytes,
                    verts_type,
                    vertex_address,
                    &["numbinds"],
                    self.header.endian,
                )?)
                .unwrap_or(0);
            let binds_num = checked_count(binds_num, "Surface Deform bind")?;
            let binds_ptr = self.schema.pointer_at_path(
                self.bytes,
                verts_type,
                vertex_address,
                &["binds"],
                self.header.endian,
                self.header.pointer_size,
            )?;
            if binds_num > 0 && binds_ptr == 0 {
                return Err("Surface Deform vertex has a bind count but no bind array".into());
            }
            let bind_size = self.schema.structs[bind_type].size;
            let bind_bytes = bind_size
                .checked_mul(binds_num)
                .ok_or_else(|| "Surface Deform bind array size overflow".to_owned())?;
            let binds_base = if binds_num == 0 {
                0
            } else {
                self.resolve(binds_ptr, bind_bytes)?.1
            };
            let bind_records = self.read_struct_array(binds_ptr, binds_num, bind_type)?;
            let mut binds = Vec::with_capacity(binds_num);
            for (bind_offset, mut bind) in bind_records.into_iter().enumerate() {
                let bind_address = checked_array_item_offset(binds_base, bind_offset, bind_size)?;
                let verts_num = self
                    .schema
                    .integer_at_path(
                        self.bytes,
                        bind_type,
                        bind_address,
                        &["verts_num"],
                        self.header.endian,
                    )?
                    .or(self.schema.integer_at_path(
                        self.bytes,
                        bind_type,
                        bind_address,
                        &["numverts"],
                        self.header.endian,
                    )?)
                    .unwrap_or(0);
                let verts_num = checked_count(verts_num, "Surface Deform bind index")?;
                let inds_ptr = self.schema.pointer_at_path(
                    self.bytes,
                    bind_type,
                    bind_address,
                    &["vert_inds"],
                    self.header.endian,
                    self.header.pointer_size,
                )?;
                let weights_ptr = self.schema.pointer_at_path(
                    self.bytes,
                    bind_type,
                    bind_address,
                    &["vert_weights"],
                    self.header.endian,
                    self.header.pointer_size,
                )?;
                if verts_num > 0 && (inds_ptr == 0 || weights_ptr == 0) {
                    return Err(
                        "Surface Deform bind has a count but missing index or weight data".into(),
                    );
                }
                if verts_num > 0 {
                    payload_found = true;
                }
                let inds_type = self
                    .schema
                    .type_names
                    .iter()
                    .position(|name| name == "unsigned int")
                    .or_else(|| {
                        self.schema
                            .type_names
                            .iter()
                            .position(|name| name == "uint")
                    })
                    .or_else(|| self.schema.type_names.iter().position(|name| name == "int"))
                    .ok_or_else(|| {
                        "Blender DNA schema does not contain an integer type".to_owned()
                    })?;
                let float_type = self
                    .schema
                    .type_names
                    .iter()
                    .position(|name| name == "float")
                    .ok_or_else(|| "Blender DNA schema does not contain float".to_owned())?;
                let inds = self.read_scalar_array(inds_ptr, verts_num, inds_type)?;
                if target_count > 0
                    && inds.iter().any(|index| {
                        json_nonnegative_index(index).is_none_or(|index| index >= target_count)
                    })
                {
                    return Err("Surface Deform bind index exceeds the target vertex count".into());
                }
                let mode = bind.get("mode").and_then(Value::as_i64).unwrap_or(-1);
                if !(0..=2).contains(&mode) {
                    return Err("Surface Deform bind mode is invalid".into());
                }
                let weight_count = if mode == 1 { verts_num } else { 3 };
                let weights = self.read_scalar_array(weights_ptr, weight_count, float_type)?;
                set_object_member(&mut bind, "vert_inds", Value::Array(inds))?;
                set_object_member(&mut bind, "vert_weights", Value::Array(weights))?;
                if bind.get("numverts").is_some() {
                    rename_object_member(&mut bind, "numverts", "verts_num")?;
                }
                binds.push(bind);
            }
            set_object_member(&mut vertex, "binds", Value::Array(binds))?;
            verts.push(vertex);
        }
        if !payload_found {
            return Ok(None);
        }
        let mut modifier = self.schema.decode_struct(
            self.bytes,
            struct_index,
            base_offset,
            self.header.endian,
            self.header.pointer_size,
            0,
        )?;
        set_object_member(&mut modifier, "verts", Value::Array(verts))?;
        rename_object_member(&mut modifier, "verts", "bind_verts")?;
        if modifier.get("num_mesh_verts").is_some() {
            rename_object_member(&mut modifier, "num_mesh_verts", "mesh_verts_num")?;
        }
        if modifier.get("numverts").is_some() {
            rename_object_member(&mut modifier, "numverts", "bind_verts_num")?;
        }
        if modifier.get("numpoly").is_some() {
            rename_object_member(&mut modifier, "numpoly", "target_polys_num")?;
        }
        set_bind_discriminator(&mut modifier, "surface_deform")?;
        Ok(Some(modifier))
    }

    fn read_mesh_payload(
        &self,
        struct_index: usize,
        base_offset: usize,
        flags: u64,
    ) -> Result<Option<Value>, String> {
        const DYNAMIC_BIND: u64 = 1 << 1;
        let verts_num = self
            .schema
            .integer_at_path(
                self.bytes,
                struct_index,
                base_offset,
                &["verts_num"],
                self.header.endian,
            )?
            .or(self.schema.integer_at_path(
                self.bytes,
                struct_index,
                base_offset,
                &["totvert"],
                self.header.endian,
            )?)
            .unwrap_or(0);
        let cage_verts_num = self
            .schema
            .integer_at_path(
                self.bytes,
                struct_index,
                base_offset,
                &["cage_verts_num"],
                self.header.endian,
            )?
            .or(self.schema.integer_at_path(
                self.bytes,
                struct_index,
                base_offset,
                &["totcagevert"],
                self.header.endian,
            )?)
            .unwrap_or(0);
        let verts_num = checked_count(verts_num, "Mesh Deform source vertex")?;
        let cage_verts_num = checked_count(cage_verts_num, "Mesh Deform cage vertex")?;
        let offsets_ptr = self.schema.optional_pointer_at_path(
            self.bytes,
            struct_index,
            base_offset,
            &["bindoffsets"],
            self.header.endian,
            self.header.pointer_size,
        )?;
        let influences_ptr = self.schema.optional_pointer_at_path(
            self.bytes,
            struct_index,
            base_offset,
            &["bindinfluences"],
            self.header.endian,
            self.header.pointer_size,
        )?;
        let cagecos_ptr = self.schema.optional_pointer_at_path(
            self.bytes,
            struct_index,
            base_offset,
            &["bindcagecos"],
            self.header.endian,
            self.header.pointer_size,
        )?;
        let dynamic = flags & DYNAMIC_BIND != 0;
        let dyngrid_ptr = self.schema.optional_pointer_at_path(
            self.bytes,
            struct_index,
            base_offset,
            &["dyngrid"],
            self.header.endian,
            self.header.pointer_size,
        )?;
        let dyninfluences_ptr = self.schema.optional_pointer_at_path(
            self.bytes,
            struct_index,
            base_offset,
            &["dyninfluences"],
            self.header.endian,
            self.header.pointer_size,
        )?;
        let dynverts_ptr = self.schema.optional_pointer_at_path(
            self.bytes,
            struct_index,
            base_offset,
            &["dynverts"],
            self.header.endian,
            self.header.pointer_size,
        )?;
        let static_bound = offsets_ptr != 0 || influences_ptr != 0 || cagecos_ptr != 0;
        let dynamic_bound =
            dynamic && (dyngrid_ptr != 0 || dyninfluences_ptr != 0 || dynverts_ptr != 0);
        if !static_bound && !dynamic_bound {
            return Ok(None);
        }

        let mut modifier = self.schema.decode_struct(
            self.bytes,
            struct_index,
            base_offset,
            self.header.endian,
            self.header.pointer_size,
            0,
        )?;
        set_object_member(&mut modifier, "dynamic_bind", Value::Bool(dynamic))?;
        let mut payload_found = false;
        if static_bound {
            if verts_num == 0 || cage_verts_num == 0 || offsets_ptr == 0 || cagecos_ptr == 0 {
                return Err(format!(
                    "bound Mesh Deform static payload is missing required arrays (source={verts_num}, cage={cage_verts_num}, offsets=0x{offsets_ptr:x}, cage_coordinates=0x{cagecos_ptr:x})"
                ));
            }
            let int_type = self
                .schema
                .type_names
                .iter()
                .position(|name| name == "int")
                .ok_or_else(|| "Blender DNA schema does not contain int".to_owned())?;
            let float_type = self
                .schema
                .type_names
                .iter()
                .position(|name| name == "float")
                .ok_or_else(|| "Blender DNA schema does not contain float".to_owned())?;
            let offsets_count = verts_num
                .checked_add(1)
                .ok_or_else(|| "Mesh Deform offset count overflow".to_owned())?;
            let offsets = self.read_scalar_array(offsets_ptr, offsets_count, int_type)?;
            if offsets.first().and_then(Value::as_i64) != Some(0) {
                return Err("Mesh Deform offsets do not start at zero".into());
            }
            let mut previous = 0_usize;
            for value in &offsets {
                let current = value
                    .as_u64()
                    .and_then(|number| usize::try_from(number).ok())
                    .or_else(|| {
                        value
                            .as_i64()
                            .and_then(|number| usize::try_from(number).ok())
                    })
                    .ok_or_else(|| "Mesh Deform offset is negative or invalid".to_owned())?;
                if current < previous || current > MAX_ARRAY_ELEMENTS {
                    return Err("Mesh Deform offsets are decreasing or excessive".into());
                }
                previous = current;
            }
            let declared_influences = self
                .schema
                .integer_at_path(
                    self.bytes,
                    struct_index,
                    base_offset,
                    &["influences_num"],
                    self.header.endian,
                )?
                .or(self.schema.integer_at_path(
                    self.bytes,
                    struct_index,
                    base_offset,
                    &["totinfluence"],
                    self.header.endian,
                )?);
            if let Some(declared_count) = declared_influences
                && checked_count(declared_count, "Mesh Deform influence")? != previous
            {
                return Err("Mesh Deform offsets disagree with the influence count".into());
            }
            if previous > 0 && influences_ptr == 0 {
                return Err("Mesh Deform offsets reference missing influence data".into());
            }
            let influence_type = self
                .schema
                .struct_index("MDefInfluence")
                .ok_or_else(|| "Blender DNA schema does not contain MDefInfluence".to_owned())?;
            let influences = self.read_struct_array(influences_ptr, previous, influence_type)?;
            for influence in &influences {
                let vertex = influence
                    .get("vertex")
                    .and_then(Value::as_i64)
                    .ok_or_else(|| "Mesh Deform influence has an invalid cage vertex".to_owned())?;
                if vertex < 0
                    || usize::try_from(vertex).map_or(true, |index| index >= cage_verts_num)
                {
                    return Err("Mesh Deform influence references an invalid cage vertex".into());
                }
            }
            let cage_count = cage_verts_num
                .checked_mul(3)
                .ok_or_else(|| "Mesh Deform cage coordinate count overflow".to_owned())?;
            let cagecos = self.read_scalar_array(cagecos_ptr, cage_count, float_type)?;
            payload_found = true;
            set_object_member(&mut modifier, "bindoffsets", Value::Array(offsets))?;
            set_object_member(&mut modifier, "bindinfluences", Value::Array(influences))?;
            set_object_member(&mut modifier, "bindcagecos", Value::Array(cagecos))?;
            rename_object_member(&mut modifier, "bindoffsets", "bind_offsets")?;
            rename_object_member(&mut modifier, "bindinfluences", "bind_influences")?;
            rename_object_member(&mut modifier, "bindcagecos", "bind_cage_coordinates")?;
        }
        if dynamic_bound {
            let dyngridsize = self
                .schema
                .integer_at_path(
                    self.bytes,
                    struct_index,
                    base_offset,
                    &["dyngridsize"],
                    self.header.endian,
                )?
                .unwrap_or(0);
            let dyngridsize = checked_count(dyngridsize, "Mesh Deform dynamic grid dimension")?;
            if dyngridsize == 0 {
                return Err("Mesh Deform dynamic bind has a zero-sized grid".into());
            }
            let grid_count = dyngridsize
                .checked_mul(dyngridsize)
                .and_then(|square| square.checked_mul(dyngridsize))
                .ok_or_else(|| "Mesh Deform dynamic grid size overflow".to_owned())?;
            if grid_count > MAX_ARRAY_ELEMENTS {
                return Err("Mesh Deform dynamic grid exceeds the element limit".into());
            }
            let influences_num = self
                .schema
                .integer_at_path(
                    self.bytes,
                    struct_index,
                    base_offset,
                    &["influences_num"],
                    self.header.endian,
                )?
                .or(self.schema.integer_at_path(
                    self.bytes,
                    struct_index,
                    base_offset,
                    &["totinfluence"],
                    self.header.endian,
                )?)
                .unwrap_or(0);
            let influences_num = checked_count(influences_num, "Mesh Deform dynamic influence")?;
            let cell_type = self
                .schema
                .struct_index("MDefCell")
                .ok_or_else(|| "Blender DNA schema does not contain MDefCell".to_owned())?;
            let influence_type = self
                .schema
                .struct_index("MDefInfluence")
                .ok_or_else(|| "Blender DNA schema does not contain MDefInfluence".to_owned())?;
            if grid_count > 0 && dyngrid_ptr == 0 {
                return Err("Mesh Deform dynamic bind has no grid data".into());
            }
            if influences_num > 0 && dyninfluences_ptr == 0 {
                return Err("Mesh Deform dynamic bind has no influence data".into());
            }
            if verts_num > 0 && dynverts_ptr == 0 {
                return Err("Mesh Deform dynamic bind has no per-vertex data".into());
            }
            let cells = self.read_struct_array(dyngrid_ptr, grid_count, cell_type)?;
            let influences =
                self.read_struct_array(dyninfluences_ptr, influences_num, influence_type)?;
            for cell in &cells {
                let offset = cell
                    .get("offset")
                    .and_then(Value::as_i64)
                    .ok_or_else(|| "Mesh Deform dynamic cell has an invalid offset".to_owned())?;
                let count = cell
                    .get("influences_num")
                    .and_then(Value::as_i64)
                    .ok_or_else(|| {
                        "Mesh Deform dynamic cell has an invalid influence count".to_owned()
                    })?;
                let end = offset
                    .checked_add(count)
                    .ok_or_else(|| "Mesh Deform dynamic cell range overflow".to_owned())?;
                if offset < 0
                    || count < 0
                    || end > i64::try_from(influences_num).unwrap_or(i64::MAX)
                {
                    return Err("Mesh Deform dynamic cell references invalid influences".into());
                }
            }
            for influence in &influences {
                let vertex = influence
                    .get("vertex")
                    .and_then(Value::as_i64)
                    .ok_or_else(|| {
                        "Mesh Deform dynamic influence has an invalid vertex".to_owned()
                    })?;
                if vertex < 0
                    || (cage_verts_num > 0
                        && usize::try_from(vertex).map_or(true, |index| index >= cage_verts_num))
                {
                    return Err("Mesh Deform dynamic influence references an invalid vertex".into());
                }
            }
            let dynverts = self.read_scalar_array(
                dynverts_ptr,
                verts_num,
                self.schema
                    .type_names
                    .iter()
                    .position(|name| name == "int")
                    .ok_or_else(|| "Blender DNA schema does not contain int".to_owned())?,
            )?;
            set_object_member(&mut modifier, "dyngrid", Value::Array(cells))?;
            set_object_member(&mut modifier, "dyninfluences", Value::Array(influences))?;
            set_object_member(&mut modifier, "dynverts", Value::Array(dynverts))?;
            rename_object_member(&mut modifier, "dyngrid", "dynamic_grid")?;
            rename_object_member(&mut modifier, "dyninfluences", "dynamic_influences")?;
            rename_object_member(&mut modifier, "dynverts", "dynamic_vertices")?;
            payload_found |= grid_count > 0 || influences_num > 0 || verts_num > 0;
        }
        if payload_found {
            rename_object_member(&mut modifier, "dyngridsize", "dynamic_grid_size")?;
            rename_object_member(&mut modifier, "dyncellmin", "dynamic_cell_min")?;
            rename_object_member(&mut modifier, "dyncellwidth", "dynamic_cell_width")?;
            rename_object_member(&mut modifier, "bindmat", "bind_matrix")?;
            if modifier.get("totvert").is_some() {
                rename_object_member(&mut modifier, "totvert", "verts_num")?;
            }
            if modifier.get("totcagevert").is_some() {
                rename_object_member(&mut modifier, "totcagevert", "cage_verts_num")?;
            }
            if modifier.get("totinfluence").is_some() {
                rename_object_member(&mut modifier, "totinfluence", "influences_num")?;
            }
            set_bind_discriminator(&mut modifier, "mesh_deform")?;
            Ok(Some(modifier))
        } else {
            Ok(None)
        }
    }

    fn read_laplacian_payload(
        &self,
        struct_index: usize,
        base_offset: usize,
        flags: u64,
    ) -> Result<Option<Value>, String> {
        const LAPLACIAN_BIND: u64 = 1;
        if flags & LAPLACIAN_BIND == 0 {
            return Ok(None);
        }
        let verts_num = self
            .schema
            .integer_at_path(
                self.bytes,
                struct_index,
                base_offset,
                &["verts_num"],
                self.header.endian,
            )?
            .or(self.schema.integer_at_path(
                self.bytes,
                struct_index,
                base_offset,
                &["total_verts"],
                self.header.endian,
            )?)
            .unwrap_or(0);
        let verts_num = checked_count(verts_num, "Laplacian Deform vertex")?;
        let pointer = self.schema.pointer_at_path(
            self.bytes,
            struct_index,
            base_offset,
            &["vertexco"],
            self.header.endian,
            self.header.pointer_size,
        )?;
        if verts_num == 0 {
            return Ok(None);
        }
        if pointer == 0 {
            return Err("bound Laplacian Deform has vertices but no coordinate array".into());
        }
        let float_count = verts_num
            .checked_mul(3)
            .ok_or_else(|| "Laplacian Deform coordinate count overflow".to_owned())?;
        let float_type = self
            .schema
            .type_names
            .iter()
            .position(|name| name == "float")
            .ok_or_else(|| "Blender DNA schema does not contain float".to_owned())?;
        let vertexco = self.read_scalar_array(pointer, float_count, float_type)?;
        let mut modifier = self.schema.decode_struct(
            self.bytes,
            struct_index,
            base_offset,
            self.header.endian,
            self.header.pointer_size,
            0,
        )?;
        set_object_member(&mut modifier, "vertexco", Value::Array(vertexco))?;
        rename_object_member(&mut modifier, "vertexco", "vertex_coordinates")?;
        if modifier.get("total_verts").is_some() {
            rename_object_member(&mut modifier, "total_verts", "verts_num")?;
        }
        set_bind_discriminator(&mut modifier, "laplacian_deform")?;
        Ok(Some(modifier))
    }

    fn resolve(&self, address: u64, length: usize) -> Result<(&Block, usize), String> {
        if address == 0 {
            return Err("null Blender pointer cannot be dereferenced".into());
        }
        for (&start, &block_index) in self.by_address.range(..=address).rev() {
            let block = &self.blocks[block_index];
            let Some(displacement) = address
                .checked_sub(start)
                .and_then(|value| usize::try_from(value).ok())
            else {
                continue;
            };
            let Some(end) = displacement.checked_add(length) else {
                continue;
            };
            if end <= block.len {
                let offset = block
                    .data_offset
                    .checked_add(displacement)
                    .ok_or_else(|| "Blender block data offset overflow".to_owned())?;
                return Ok((block, offset));
            }
        }
        Err(format!(
            "Blender pointer 0x{address:x} does not fit a data block"
        ))
    }
    fn read_scalar_array(
        &self,
        pointer: u64,
        count: usize,
        type_index: usize,
    ) -> Result<Vec<Value>, String> {
        if count > MAX_ARRAY_ELEMENTS {
            return Err("Blender pointer array exceeds the element limit".into());
        }
        if count == 0 {
            return Ok(Vec::new());
        }
        let size = *self
            .schema
            .type_lengths
            .get(type_index)
            .ok_or_else(|| "Blender pointer array type index is invalid".to_owned())?;
        if size == 0 {
            return Err("zero-sized Blender scalar array element".into());
        }
        let byte_count = size
            .checked_mul(count)
            .ok_or_else(|| "Blender pointer array byte size overflow".to_owned())?;
        let (block, offset) = self.resolve(pointer, byte_count)?;
        if (offset - block.data_offset) % size != 0 {
            return Err("Blender scalar-array pointer is misaligned".into());
        }
        (0..count)
            .map(|index| {
                let address = offset
                    .checked_add(
                        index
                            .checked_mul(size)
                            .ok_or_else(|| "Blender pointer array offset overflow".to_owned())?,
                    )
                    .ok_or_else(|| "Blender pointer array offset overflow".to_owned())?;
                self.schema.decode_type(
                    self.bytes,
                    type_index,
                    address,
                    self.header.endian,
                    self.header.pointer_size,
                    0,
                )
            })
            .collect()
    }

    fn read_struct_array(
        &self,
        pointer: u64,
        count: usize,
        struct_index: usize,
    ) -> Result<Vec<Value>, String> {
        if count > MAX_ARRAY_ELEMENTS {
            return Err("Blender pointer array exceeds the element limit".into());
        }
        if count == 0 {
            return Ok(Vec::new());
        }
        let size = self.schema.structs[struct_index].size;
        if size == 0 {
            return Err("zero-sized Blender structure array element".into());
        }
        let byte_count = size
            .checked_mul(count)
            .ok_or_else(|| "Blender structure array byte size overflow".to_owned())?;
        let (block, offset) = self.resolve(pointer, byte_count)?;
        if (offset - block.data_offset) % size != 0 {
            return Err("Blender structure-array pointer is misaligned".into());
        }
        (0..count)
            .map(|index| {
                let address = offset
                    .checked_add(
                        index
                            .checked_mul(size)
                            .ok_or_else(|| "Blender structure array offset overflow".to_owned())?,
                    )
                    .ok_or_else(|| "Blender structure array offset overflow".to_owned())?;
                self.schema.decode_struct(
                    self.bytes,
                    struct_index,
                    address,
                    self.header.endian,
                    self.header.pointer_size,
                    0,
                )
            })
            .collect()
    }
}

fn checked_array_item_offset(
    base: usize,
    index: usize,
    element_size: usize,
) -> Result<usize, String> {
    base.checked_add(
        index
            .checked_mul(element_size)
            .ok_or_else(|| "Blender structure array offset overflow".to_owned())?,
    )
    .ok_or_else(|| "Blender structure array offset overflow".to_owned())
}
#[derive(Clone, Copy)]
enum ModifierKind {
    Surface,
    Mesh,
    Laplacian,
}

fn checked_count(value: i64, label: &str) -> Result<usize, String> {
    let count = usize::try_from(value).map_err(|_| format!("negative {label} count"))?;
    if count > MAX_ARRAY_ELEMENTS {
        return Err(format!("{label} count exceeds the limit"));
    }
    Ok(count)
}
fn json_nonnegative_index(value: &Value) -> Option<usize> {
    value
        .as_u64()
        .and_then(|number| usize::try_from(number).ok())
        .or_else(|| {
            value
                .as_i64()
                .and_then(|number| usize::try_from(number).ok())
        })
}

fn set_object_member(object: &mut Value, key: &str, value: Value) -> Result<(), String> {
    let map = object
        .as_object_mut()
        .ok_or_else(|| "decoded Blender DNA value is not an object".to_owned())?;
    map.insert(key.to_owned(), value);
    Ok(())
}
fn rename_object_member(object: &mut Value, old_name: &str, new_name: &str) -> Result<(), String> {
    let map = object
        .as_object_mut()
        .ok_or_else(|| "decoded Blender DNA value is not an object".to_owned())?;
    if let Some(value) = map.remove(old_name) {
        map.insert(new_name.to_owned(), value);
    }
    Ok(())
}

fn set_bind_discriminator(object: &mut Value, modifier_type: &str) -> Result<(), String> {
    set_object_member(
        object,
        "format",
        Value::String("blender_native_bind_v1".to_owned()),
    )?;
    set_object_member(object, "type", Value::String(modifier_type.to_owned()))
}

fn parse_blocks(bytes: &[u8], header: Header) -> Result<Vec<Block>, String> {
    let block_header_size = if header.large_bhead {
        32
    } else {
        4_usize
            .checked_add(4)
            .and_then(|size| size.checked_add(header.pointer_size))
            .and_then(|size| size.checked_add(4 + 4))
            .ok_or_else(|| "Blender block header size overflow".to_owned())?
    };
    let mut position = header.preamble_len;
    let mut blocks = Vec::new();
    let mut found_end = false;
    while position < bytes.len() {
        if blocks.len() >= MAX_BLOCKS {
            return Err("Blender file contains too many blocks".into());
        }
        let header_bytes = checked_slice(bytes, position, block_header_size)?;
        let code: [u8; 4] = header_bytes[..4]
            .try_into()
            .map_err(|_| "invalid Blender block code".to_owned())?;
        let (sdna_start, old_start, len, count) = if header.large_bhead {
            let length = i64::try_from(header.endian.read_u64(&header_bytes[16..24])?)
                .map_err(|_| "negative or excessive Blender block length".to_owned())?;
            let count = i64::try_from(header.endian.read_u64(&header_bytes[24..32])?)
                .map_err(|_| "negative or excessive Blender block element count".to_owned())?;
            (
                4,
                8,
                usize::try_from(length)
                    .map_err(|_| "Blender block length exceeds platform limits".to_owned())?,
                usize::try_from(count).map_err(|_| {
                    "Blender block element count exceeds platform limits".to_owned()
                })?,
            )
        } else {
            let length = usize::try_from(header.endian.read_u32(&header_bytes[4..8])?)
                .map_err(|_| "Blender block length exceeds platform limits".to_owned())?;
            let old_start = 8;
            let sdna_start = 8 + header.pointer_size;
            let count = usize::try_from(
                header
                    .endian
                    .read_u32(&header_bytes[sdna_start + 4..sdna_start + 8])?,
            )
            .map_err(|_| "Blender block element count exceeds platform limits".to_owned())?;
            (sdna_start, old_start, length, count)
        };
        let old_address =
            header.read_pointer(&header_bytes[old_start..old_start + header.pointer_size])?;
        let sdna_index = usize::try_from(
            header
                .endian
                .read_u32(&header_bytes[sdna_start..sdna_start + 4])?,
        )
        .map_err(|_| "Blender block SDNA index exceeds platform limits".to_owned())?;
        let data_offset = position
            .checked_add(block_header_size)
            .ok_or_else(|| "Blender block data offset overflow".to_owned())?;
        checked_slice(bytes, data_offset, len)?;
        if &code == b"ENDB" {
            if len != 0 {
                return Err("Blender ENDB block must be empty".into());
            }
            found_end = true;
            if data_offset != bytes.len() {
                return Err("unexpected trailing data after Blender ENDB block".into());
            }
            break;
        }
        blocks.push(Block {
            code,
            len,
            old_address,
            sdna_index,
            count,
            data_offset,
        });
        position = data_offset
            .checked_add(len)
            .ok_or_else(|| "Blender block end offset overflow".to_owned())?;
    }
    if !found_end {
        return Err(format!(
            "Blender file is missing its ENDB block at byte {position} of {}",
            bytes.len()
        ));
    }
    Ok(blocks)
}

#[cfg(test)]
mod tests {
    use super::{BlendFile, Endian, Header, Schema, parse_blocks};

    fn must<T>(result: Result<T, String>) -> T {
        match result {
            Ok(value) => value,
            Err(message) => panic!("unexpected parser error: {message}"),
        }
    }

    fn push_u16(bytes: &mut Vec<u8>, value: u16) {
        bytes.extend_from_slice(&value.to_le_bytes());
    }

    fn push_u32(bytes: &mut Vec<u8>, value: u32) {
        bytes.extend_from_slice(&value.to_le_bytes());
    }

    fn push_c_string(bytes: &mut Vec<u8>, value: &str) {
        bytes.extend_from_slice(value.as_bytes());
        bytes.push(0);
    }

    fn align4(bytes: &mut Vec<u8>) {
        while !bytes.len().is_multiple_of(4) {
            bytes.push(0);
        }
    }

    fn sample_dna() -> Vec<u8> {
        let mut dna = Vec::new();
        dna.extend_from_slice(b"SDNA");
        dna.extend_from_slice(b"NAME");
        push_u32(&mut dna, 2);
        push_c_string(&mut dna, "name[3]");
        push_c_string(&mut dna, "value");
        align4(&mut dna);

        dna.extend_from_slice(b"TYPE");
        push_u32(&mut dna, 3);
        push_c_string(&mut dna, "char");
        push_c_string(&mut dna, "int");
        push_c_string(&mut dna, "Sample");
        align4(&mut dna);

        dna.extend_from_slice(b"TLEN");
        push_u16(&mut dna, 1);
        push_u16(&mut dna, 4);
        push_u16(&mut dna, 7);
        align4(&mut dna);

        dna.extend_from_slice(b"STRC");
        push_u32(&mut dna, 1);
        push_u16(&mut dna, 2);
        push_u16(&mut dna, 2);
        push_u16(&mut dna, 0);
        push_u16(&mut dna, 0);
        push_u16(&mut dna, 1);
        push_u16(&mut dna, 1);
        dna
    }

    fn append_block(bytes: &mut Vec<u8>, code: [u8; 4], data: &[u8], pointer_size: usize) {
        append_block_at(bytes, code, data, pointer_size, 0);
    }

    fn append_block_at(
        bytes: &mut Vec<u8>,
        code: [u8; 4],
        data: &[u8],
        pointer_size: usize,
        old_address: u64,
    ) {
        bytes.extend_from_slice(&code);
        push_u32(bytes, u32::try_from(data.len()).unwrap_or(u32::MAX));
        if pointer_size == 4 {
            push_u32(bytes, u32::try_from(old_address).unwrap_or(u32::MAX));
        } else {
            bytes.extend_from_slice(&old_address.to_le_bytes());
        }
        push_u32(bytes, 0);
        push_u32(bytes, 1);
        bytes.extend_from_slice(data);
    }
    fn sample_file(pointer_size: usize) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"BLENDER");
        bytes.push(if pointer_size == 4 { b'_' } else { b'-' });
        bytes.push(b'v');
        bytes.extend_from_slice(b"300");
        append_block(&mut bytes, *b"DNA1", &sample_dna(), pointer_size);
        append_block_at(
            &mut bytes,
            *b"DATA",
            &[1, 2, 3, 4, 5, 6, 7],
            pointer_size,
            0x1000,
        );
        append_block(&mut bytes, *b"ENDB", &[], pointer_size);
        bytes
    }

    fn append_large_block(
        bytes: &mut Vec<u8>,
        code: [u8; 4],
        data: &[u8],
        sdna_index: u32,
        old_address: u64,
        count: u64,
    ) {
        bytes.extend_from_slice(&code);
        push_u32(bytes, sdna_index);
        bytes.extend_from_slice(&old_address.to_le_bytes());
        bytes.extend_from_slice(&u64::try_from(data.len()).unwrap_or(u64::MAX).to_le_bytes());
        bytes.extend_from_slice(&count.to_le_bytes());
        bytes.extend_from_slice(data);
    }

    fn sample_large_header_file() -> Vec<u8> {
        let mut bytes = b"BLENDER17-01v0502".to_vec();
        append_large_block(&mut bytes, *b"DNA1", &sample_dna(), 0, 0, 1);
        append_large_block(&mut bytes, *b"DATA", &[1, 2, 3, 4, 5, 6, 7], 0, 0x1000, 1);
        append_large_block(&mut bytes, *b"ENDB", &[], 0, 0, 0);
        bytes
    }

    #[test]
    fn parses_sdna_names_and_member_offsets() {
        let schema = must(Schema::parse(&sample_dna(), Endian::Little, 8));
        let Some(sample) = schema.struct_index("Sample") else {
            panic!("Sample type was not parsed");
        };
        assert_eq!(schema.structs[sample].size, 7);
        let Some(name) = schema.field(sample, "name") else {
            panic!("name field was not parsed");
        };
        let Some(value) = schema.field(sample, "value") else {
            panic!("value field was not parsed");
        };
        assert_eq!(name.offset, 0);
        assert_eq!(name.size, 3);
        assert_eq!(value.offset, 3);
        assert_eq!(value.size, 4);
    }

    #[test]
    fn parses_bhead_blocks_for_32_and_64_bit_headers() {
        for pointer_size in [4, 8] {
            let bytes = sample_file(pointer_size);
            let header = must(Header::parse(&bytes));
            assert_eq!(header.pointer_size, pointer_size);
            assert!(matches!(header.endian, Endian::Little));
            let blocks = must(parse_blocks(&bytes, header));
            assert_eq!(blocks.len(), 2);
            assert_eq!(&blocks[0].code, b"DNA1");
            assert_eq!(&blocks[1].code, b"DATA");
            assert_eq!(blocks[0].len, sample_dna().len());
            let parsed = must(BlendFile::parse(&bytes));
            assert!(parsed.schema.struct_index("Sample").is_some());
            let (data_block, offset) = must(parsed.resolve(0x1002, 3));
            assert_eq!(&parsed.bytes[offset..offset + 3], &[3, 4, 5]);
            assert_eq!(data_block.old_address, 0x1000);
        }
    }

    #[test]
    fn parses_blender_large_header_and_large_bheads() {
        let bytes = sample_large_header_file();
        let header = must(Header::parse(&bytes));
        assert_eq!(header.pointer_size, 8);
        assert_eq!(header.preamble_len, 17);
        assert!(header.large_bhead);
        assert!(matches!(header.endian, Endian::Little));
        let blocks = must(parse_blocks(&bytes, header));
        assert_eq!(blocks.len(), 2);
        assert_eq!(&blocks[0].code, b"DNA1");
        assert_eq!(&blocks[1].code, b"DATA");
        let parsed = must(BlendFile::parse(&bytes));
        let (data_block, offset) = must(parsed.resolve(0x1002, 3));
        assert_eq!(&parsed.bytes[offset..offset + 3], &[3, 4, 5]);
        assert_eq!(data_block.old_address, 0x1000);
    }

    #[test]
    fn parses_big_endian_header_marker() {
        let mut bytes = sample_file(8);
        bytes[8] = b'V';
        let header = must(Header::parse(&bytes));
        assert!(matches!(header.endian, Endian::Big));
        assert_eq!(
            Endian::Big.read_u32(&[0x01, 0x02, 0x03, 0x04]),
            Ok(0x0102_0304)
        );
    }

    #[test]
    fn rejects_truncated_bhead_data() {
        let mut bytes = sample_file(8);
        bytes.truncate(bytes.len() - 1);
        assert!(parse_blocks(&bytes, must(Header::parse(&bytes))).is_err());
    }
}
