use std::{fs, path::Path};

use serde_json::{Map, Value, json};

use crate::{
    error::{ErrorCode, PotError, Result},
    hash,
    image::{ImageInterpolation, encode_pixels},
    model::{Id, Image, ImageAlphaMode, ImageColorspace, ImageSource, ImageTile},
};

use super::{
    ChangeKind, Engine, check_fields, operation_pointer, parse_id, pointer_escape, read_id,
};

pub(super) fn apply(
    engine: &mut Engine<'_>,
    name: &str,
    operation: &Map<String, Value>,
) -> Result<bool> {
    match name {
        "image.create" => create(engine, operation),
        "image.load" => load(engine, operation),
        "image.set_pixels" => set_pixels(engine, operation),
        "image.update" => update(engine, operation),
        "image.delete" => delete(engine, operation),
        _ => Err(engine.error(
            ErrorCode::InternalError,
            "invalid image operation dispatch",
            &operation_pointer(engine.operation_index, "op"),
        )),
    }
}

fn create(engine: &mut Engine<'_>, operation: &Map<String, Value>) -> Result<bool> {
    check_fields(
        engine,
        operation,
        &[
            "op",
            "id",
            "name",
            "width",
            "height",
            "colorspace",
            "fill_color",
            "generator",
            "checker_colors",
            "checker_size",
            "alpha_mode",
        ],
        &["id", "width", "height"],
    )?;
    let id = read_id(engine, operation, "id")?;
    if engine.doc.images.contains_key(&id) {
        return Err(engine.error(
            ErrorCode::IdExists,
            format!("image ID `{id}` already exists"),
            &operation_pointer(engine.operation_index, "id"),
        ));
    }
    let width = read_dimension(engine, operation, "width")?;
    let height = read_dimension(engine, operation, "height")?;
    validate_dimensions(engine, width, height)?;
    let pixel_count = pixel_count(width, height)?;
    let colorspace =
        parse_colorspace(engine, operation.get("colorspace"), ImageColorspace::Linear)?;
    let generator = match operation.get("generator") {
        None => "solid",
        Some(Value::String(generator)) => generator.as_str(),
        Some(_) => {
            return Err(engine.error(
                ErrorCode::InvalidOperation,
                "generator must be a string",
                &operation_pointer(engine.operation_index, "generator"),
            ));
        }
    };
    if !["solid", "checker"].contains(&generator) {
        return Err(engine.error(
            ErrorCode::InvalidOperation,
            "generator must be solid or checker",
            &operation_pointer(engine.operation_index, "generator"),
        ));
    }
    let fill = read_color(
        engine,
        operation.get("fill_color"),
        [0.0, 0.0, 0.0, 0.0],
        "fill_color",
    )?;
    let (first, second) = if generator == "checker" {
        read_checker_colors(engine, operation.get("checker_colors"))?
    } else {
        (fill, fill)
    };
    let checker_size = operation
        .get("checker_size")
        .map(|value| read_positive_integer(engine, value, "checker_size"))
        .transpose()?
        .unwrap_or(8);
    if operation
        .get("alpha_mode")
        .is_some_and(|value| !matches!(value.as_str(), Some("straight" | "premultiplied" | "none")))
    {
        return Err(engine.error(
            ErrorCode::InvalidOperation,
            "alpha_mode must be straight, premultiplied, or none",
            &operation_pointer(engine.operation_index, "alpha_mode"),
        ));
    }
    let width_usize = usize::try_from(width).map_err(|_| {
        PotError::new(
            ErrorCode::LimitExceeded,
            "image width exceeds platform limits",
        )
    })?;
    let pixels = (0..pixel_count)
        .map(|index| {
            let x = index % width_usize;
            let y = index / width_usize;
            if generator == "checker" && ((x / checker_size) + (y / checker_size)) % 2 == 1 {
                second
            } else {
                first
            }
        })
        .collect::<Vec<_>>();
    let bytes = encode_pixels(width, height, &pixels)?;
    let blob = stage_blob(engine, bytes);
    let alpha_mode = operation.get("alpha_mode").and_then(Value::as_str).map_or(
        ImageAlphaMode::Straight,
        |value| match value {
            "premultiplied" => ImageAlphaMode::Premultiplied,
            "none" => ImageAlphaMode::None,
            _ => ImageAlphaMode::Straight,
        },
    );
    let name = operation
        .get("name")
        .map(|_| super::read_string(engine, operation, "name"))
        .transpose()?
        .unwrap_or_else(|| id.to_string());
    engine.doc.images.insert(
        id.clone(),
        Image {
            name,
            source: ImageSource::Generated,
            colorspace,
            width,
            height,
            tiles: Vec::new(),
            blob: Some(blob),
            source_path: None,
            source_hash: None,
            alpha_mode,
        },
    );
    engine.mark("images", &id, ChangeKind::Created);
    Ok(true)
}

fn load(engine: &mut Engine<'_>, operation: &Map<String, Value>) -> Result<bool> {
    check_fields(
        engine,
        operation,
        &["op", "id", "name", "path", "colorspace", "asset_policy"],
        &["id", "path"],
    )?;
    let id = read_id(engine, operation, "id")?;
    if engine.doc.images.contains_key(&id) {
        return Err(engine.error(
            ErrorCode::IdExists,
            format!("image ID `{id}` already exists"),
            &operation_pointer(engine.operation_index, "id"),
        ));
    }
    let source = super::read_string(engine, operation, "path")?;
    let source_path = Path::new(&source);
    let canonical_path = fs::canonicalize(source_path).map_err(|error| {
        if error.kind() == std::io::ErrorKind::NotFound {
            PotError::with_details(
                ErrorCode::FileNotFound,
                "image source file was not found",
                json!({"path":source,"operation_index":engine.operation_index,"pointer":operation_pointer(engine.operation_index,"path")}),
            )
        } else {
            PotError::io(&error)
        }
    })?;
    let bytes = fs::read(&canonical_path).map_err(|error| PotError::io(&error))?;
    let requested_colorspace = operation
        .get("colorspace")
        .map(|value| parse_colorspace(engine, Some(value), ImageColorspace::Srgb))
        .transpose()?;
    let (colorspace, width, height, _) =
        crate::image::storage::decode_pixels_with_default_colorspace(&bytes, requested_colorspace)?;
    validate_dimensions(engine, width, height)?;
    let asset_policy = match operation.get("asset_policy") {
        None => "copy",
        Some(Value::String(policy)) => policy.as_str(),
        Some(_) => {
            return Err(engine.error(
                ErrorCode::InvalidOperation,
                "asset_policy must be a string",
                &operation_pointer(engine.operation_index, "asset_policy"),
            ));
        }
    };
    if !["copy", "link"].contains(&asset_policy) {
        return Err(engine.error(
            ErrorCode::InvalidOperation,
            "asset_policy must be copy or link",
            &operation_pointer(engine.operation_index, "asset_policy"),
        ));
    }
    let source_hash = hash::sha256(&bytes);
    let (image_source, blob, image_source_path, registered_hash) = if asset_policy == "copy" {
        let blob = stage_blob(engine, bytes);
        (ImageSource::Packed, Some(blob), None, Some(source_hash))
    } else {
        (
            ImageSource::File,
            None,
            Some(canonical_path.to_string_lossy().into_owned()),
            Some(source_hash),
        )
    };
    let name = operation
        .get("name")
        .map(|_| super::read_string(engine, operation, "name"))
        .transpose()?
        .unwrap_or_else(|| {
            canonical_path
                .file_name()
                .and_then(|name| name.to_str())
                .unwrap_or(id.as_str())
                .to_owned()
        });
    engine.doc.images.insert(
        id.clone(),
        Image {
            name,
            source: image_source,
            colorspace,
            width,
            height,
            tiles: Vec::new(),
            blob,
            source_path: image_source_path,
            source_hash: registered_hash,
            alpha_mode: ImageAlphaMode::Straight,
        },
    );
    engine.mark("images", &id, ChangeKind::Created);
    Ok(true)
}

fn set_pixels(engine: &mut Engine<'_>, operation: &Map<String, Value>) -> Result<bool> {
    check_fields(
        engine,
        operation,
        &[
            "op", "id", "image", "target", "tile", "x", "y", "width", "height", "pixels",
        ],
        &["pixels"],
    )?;
    let id = operation_image_id(engine, operation)?;
    let original = engine.doc.images.get(&id).cloned().ok_or_else(|| {
        engine.error(
            ErrorCode::TargetNotFound,
            format!("image `{id}` was not found"),
            &operation_pointer(engine.operation_index, "target"),
        )
    })?;
    let root = engine.asset_root.ok_or_else(|| {
        engine.error(
            ErrorCode::InvalidOperation,
            "image.set_pixels requires a project asset context",
            &operation_pointer(engine.operation_index, "op"),
        )
    })?;
    let mut data = crate::image::load_image_data_with_staged(
        &original,
        root,
        ImageInterpolation::Closest,
        &engine.pending_assets,
    )?;
    let tile_number = operation
        .get("tile")
        .map(|value| read_positive_u32(engine, value, "tile"))
        .transpose()?
        .unwrap_or(1001);
    if tile_number < 1001 {
        return Err(engine.error(
            ErrorCode::InvalidOperation,
            "UDIM tile number must be at least 1001",
            &operation_pointer(engine.operation_index, "tile"),
        ));
    }
    let (tile_width, tile_height) = if tile_number == 1001 {
        (data.width, data.height)
    } else if let Some(tile) = data.tiles.get(&tile_number) {
        (tile.width, tile.height)
    } else {
        let width = read_dimension(engine, operation, "width")?;
        let height = read_dimension(engine, operation, "height")?;
        validate_dimensions(engine, width, height)?;
        let pixel_count = pixel_count(width, height)?;
        data.tiles.insert(
            tile_number,
            crate::image::ImageTileData {
                width,
                height,
                pixels: vec![[0.0; 4]; pixel_count],
            },
        );
        (width, height)
    };
    let pixels = parse_pixels(engine, operation.get("pixels"))?;
    let x = optional_dimension(engine, operation, "x", 0)?;
    let y = optional_dimension(engine, operation, "y", 0)?;
    let width = operation
        .get("width")
        .map(|value| read_dimension_value(engine, value, "width"))
        .transpose()?
        .unwrap_or(tile_width);
    let height = operation
        .get("height")
        .map(|value| read_dimension_value(engine, value, "height"))
        .transpose()?
        .unwrap_or(tile_height);
    validate_dimensions(engine, width, height)?;
    if pixel_count(width, height)? != pixels.len() {
        return Err(engine.error(
            ErrorCode::InvalidOperation,
            "pixels length must match width times height",
            &operation_pointer(engine.operation_index, "pixels"),
        ));
    }
    let end_x = x.checked_add(width);
    let end_y = y.checked_add(height);
    if end_x.is_none_or(|end| end > tile_width) || end_y.is_none_or(|end| end > tile_height) {
        return Err(engine.error(
            ErrorCode::InvalidOperation,
            "pixel region is outside the image tile",
            &operation_pointer(engine.operation_index, "pixels"),
        ));
    }
    let target_pixels = if tile_number == 1001 {
        &mut data.pixels
    } else {
        &mut data
            .tiles
            .get_mut(&tile_number)
            .ok_or_else(|| {
                PotError::new(ErrorCode::InternalError, "created UDIM tile disappeared")
            })?
            .pixels
    };
    let tile_stride = usize::try_from(tile_width).map_err(|_| {
        PotError::new(
            ErrorCode::LimitExceeded,
            "image width exceeds platform limits",
        )
    })?;
    let region_stride = usize::try_from(width).map_err(|_| {
        PotError::new(
            ErrorCode::LimitExceeded,
            "pixel region width exceeds platform limits",
        )
    })?;
    let offset_x = usize::try_from(x).map_err(|_| {
        PotError::new(
            ErrorCode::LimitExceeded,
            "pixel offset exceeds platform limits",
        )
    })?;
    let offset_y = usize::try_from(y).map_err(|_| {
        PotError::new(
            ErrorCode::LimitExceeded,
            "pixel offset exceeds platform limits",
        )
    })?;
    let height_usize = usize::try_from(height).map_err(|_| {
        PotError::new(
            ErrorCode::LimitExceeded,
            "pixel region height exceeds platform limits",
        )
    })?;
    let mut changed = false;
    for row in 0..height_usize {
        let target_start = (offset_y + row) * tile_stride + offset_x;
        let source_start = row * region_stride;
        for column in 0..region_stride {
            let before = target_pixels[target_start + column];
            let after = pixels[source_start + column];
            if !crate::float::equal_f64_array(&before, &after) {
                target_pixels[target_start + column] = after;
                changed = true;
            }
        }
    }
    if !changed {
        return Ok(false);
    }
    let mut updated = original;
    if tile_number == 1001 {
        let bytes = encode_pixels(data.width, data.height, &data.pixels)?;
        updated.blob = Some(stage_blob(engine, bytes));
        updated.width = data.width;
        updated.height = data.height;
    } else {
        let tile = data.tiles.get(&tile_number).ok_or_else(|| {
            PotError::new(
                ErrorCode::InternalError,
                "UDIM tile disappeared while storing pixels",
            )
        })?;
        let bytes = encode_pixels(tile.width, tile.height, &tile.pixels)?;
        let blob = stage_blob(engine, bytes);
        if let Some(metadata) = updated
            .tiles
            .iter_mut()
            .find(|tile| tile.number == tile_number)
        {
            metadata.width = tile.width;
            metadata.height = tile.height;
            metadata.blob = blob;
        } else {
            updated.tiles.push(ImageTile {
                number: tile_number,
                width: tile.width,
                height: tile.height,
                blob,
            });
            updated.tiles.sort_by_key(|tile| tile.number);
        }
    }
    if updated.blob.is_none() {
        let bytes = encode_pixels(data.width, data.height, &data.pixels)?;
        updated.blob = Some(stage_blob(engine, bytes));
    }
    updated.source = ImageSource::Packed;
    updated.source_path = None;
    updated.source_hash = None;
    engine.doc.images.insert(id.clone(), updated);
    engine.mark("images", &id, ChangeKind::Updated);
    Ok(true)
}

fn update(engine: &mut Engine<'_>, operation: &Map<String, Value>) -> Result<bool> {
    check_fields(engine, operation, &["op", "id", "target", "set"], &[])?;
    let id = operation_image_id(engine, operation)?;
    let set = operation
        .get("set")
        .and_then(Value::as_object)
        .ok_or_else(|| {
            engine.error(
                ErrorCode::InvalidOperation,
                "set must be an object",
                &operation_pointer(engine.operation_index, "set"),
            )
        })?;
    if set.is_empty() {
        return Err(engine.error(
            ErrorCode::InvalidOperation,
            "set must not be empty",
            &operation_pointer(engine.operation_index, "set"),
        ));
    }
    for key in set.keys() {
        if !["name", "colorspace"].contains(&key.as_str()) {
            return Err(engine.error(
                ErrorCode::InvalidOperation,
                format!("unknown image set field `{key}`"),
                &operation_pointer(
                    engine.operation_index,
                    &format!("set/{}", pointer_escape(key)),
                ),
            ));
        }
    }
    let name = set
        .get("name")
        .map(|value| {
            value.as_str().map(str::to_owned).ok_or_else(|| {
                engine.error(
                    ErrorCode::InvalidOperation,
                    "name must be a string",
                    &operation_pointer(engine.operation_index, "set/name"),
                )
            })
        })
        .transpose()?;
    let colorspace = set
        .get("colorspace")
        .map(|value| parse_colorspace(engine, Some(value), ImageColorspace::Linear))
        .transpose()?;
    let Some(image) = engine.doc.images.get_mut(&id) else {
        return Err(engine.error(
            ErrorCode::TargetNotFound,
            format!("image `{id}` was not found"),
            &operation_pointer(engine.operation_index, "target"),
        ));
    };
    let mut changed = false;
    if let Some(name) = name
        && image.name != name
    {
        image.name = name;
        changed = true;
    }
    if let Some(colorspace) = colorspace
        && image.colorspace != colorspace
    {
        image.colorspace = colorspace;
        changed = true;
    }
    if changed {
        engine.mark("images", &id, ChangeKind::Updated);
    }
    Ok(changed)
}

fn delete(engine: &mut Engine<'_>, operation: &Map<String, Value>) -> Result<bool> {
    check_fields(engine, operation, &["op", "id", "target"], &[])?;
    let id = operation_image_id(engine, operation)?;
    if !engine.doc.images.contains_key(&id) {
        return Err(engine.error(
            ErrorCode::TargetNotFound,
            format!("image `{id}` was not found"),
            &operation_pointer(engine.operation_index, "target"),
        ));
    }
    let material_references = engine
        .doc
        .materials
        .iter()
        .filter_map(|(material_id, material)| {
            [
                material.base_color_texture.as_ref(),
                material.roughness_texture.as_ref(),
                material.metallic_texture.as_ref(),
                material.normal_texture.as_ref(),
            ]
            .into_iter()
            .flatten()
            .any(|texture| texture.image == id)
            .then_some(material_id.to_string())
        })
        .collect::<Vec<_>>();
    let graph_references = engine
        .doc
        .node_groups
        .iter()
        .filter_map(|(group_id, group)| {
            serde_json::to_value(group)
                .ok()
                .filter(|value| contains_image_reference(value, id.as_str()))
                .map(|_| group_id.to_string())
        })
        .collect::<Vec<_>>();
    let resource_references = engine
        .doc
        .resources
        .iter()
        .filter_map(|(resource_id, resource)| {
            contains_image_reference(resource, id.as_str()).then_some(resource_id.to_string())
        })
        .collect::<Vec<_>>();
    if !material_references.is_empty()
        || !graph_references.is_empty()
        || !resource_references.is_empty()
    {
        return Err(PotError::with_details(
            ErrorCode::InvalidOperation,
            "image is still referenced by a material, shader graph, or resource",
            json!({"operation_index":engine.operation_index,"pointer":operation_pointer(engine.operation_index,"target"),"image":id,"materials":material_references,"node_groups":graph_references,"resources":resource_references}),
        ));
    }
    engine.doc.images.remove(&id);
    engine.mark("images", &id, ChangeKind::Deleted);
    Ok(true)
}

fn contains_image_reference(value: &Value, id: &str) -> bool {
    match value {
        Value::Object(object) => object.iter().any(|(key, child)| {
            (matches!(key.as_str(), "image" | "image_id") && child.as_str() == Some(id))
                || contains_image_reference(child, id)
        }),
        Value::Array(values) => values
            .iter()
            .any(|child| contains_image_reference(child, id)),
        _ => false,
    }
}

fn operation_image_id(engine: &Engine<'_>, operation: &Map<String, Value>) -> Result<Id> {
    if operation.contains_key("id") {
        return read_id(engine, operation, "id");
    }
    if operation.contains_key("image") {
        return read_id(engine, operation, "image");
    }
    let target = operation
        .get("target")
        .and_then(Value::as_object)
        .ok_or_else(|| {
            engine.error(
                ErrorCode::InvalidOperation,
                "image operation requires id or target",
                &operation_pointer(engine.operation_index, "target"),
            )
        })?;
    if target.len() != 1 || !target.contains_key("id") {
        return Err(engine.error(
            ErrorCode::InvalidOperation,
            "image target must be {id}",
            &operation_pointer(engine.operation_index, "target"),
        ));
    }
    let value = target.get("id").and_then(Value::as_str).ok_or_else(|| {
        engine.error(
            ErrorCode::InvalidOperation,
            "target id must be a string",
            &operation_pointer(engine.operation_index, "target/id"),
        )
    })?;
    parse_id(
        engine,
        value,
        &operation_pointer(engine.operation_index, "target/id"),
    )
}

fn parse_colorspace(
    engine: &Engine<'_>,
    value: Option<&Value>,
    default: ImageColorspace,
) -> Result<ImageColorspace> {
    match value {
        None => Ok(default),
        Some(Value::String(value)) => match value.as_str() {
            "srgb" => Ok(ImageColorspace::Srgb),
            "linear" => Ok(ImageColorspace::Linear),
            "non_color" => Ok(ImageColorspace::NonColor),
            _ => Err(engine.error(
                ErrorCode::InvalidOperation,
                "colorspace must be srgb, linear, or non_color",
                &operation_pointer(engine.operation_index, "colorspace"),
            )),
        },
        Some(_) => Err(engine.error(
            ErrorCode::InvalidOperation,
            "colorspace must be a string",
            &operation_pointer(engine.operation_index, "colorspace"),
        )),
    }
}

fn read_color(
    engine: &Engine<'_>,
    value: Option<&Value>,
    default: [f64; 4],
    field: &str,
) -> Result<[f64; 4]> {
    let Some(value) = value else {
        return Ok(default);
    };
    let components = value
        .as_array()
        .filter(|components| components.len() == 4)
        .ok_or_else(|| {
            engine.error(
                ErrorCode::InvalidOperation,
                "pixel color must have four RGBA components",
                &operation_pointer(engine.operation_index, field),
            )
        })?;
    let [r, g, b, a] = [0, 1, 2, 3].map(|index| {
        components[index]
            .as_f64()
            .filter(|component| component.is_finite())
    });
    match [r, g, b, a] {
        [Some(r), Some(g), Some(b), Some(a)] => Ok([r, g, b, a]),
        _ => Err(engine.error(
            ErrorCode::InvalidOperation,
            "pixel components must be finite numbers",
            &operation_pointer(engine.operation_index, field),
        )),
    }
}

fn read_checker_colors(engine: &Engine<'_>, value: Option<&Value>) -> Result<([f64; 4], [f64; 4])> {
    let Some(values) = value else {
        return Ok(([1.0; 4], [0.0, 0.0, 0.0, 1.0]));
    };
    let colors = values
        .as_array()
        .filter(|colors| colors.len() == 2)
        .ok_or_else(|| {
            engine.error(
                ErrorCode::InvalidOperation,
                "checker_colors must contain two RGBA colors",
                &operation_pointer(engine.operation_index, "checker_colors"),
            )
        })?;
    Ok((
        read_color(engine, colors.first(), [0.0; 4], "checker_colors/0")?,
        read_color(engine, colors.get(1), [0.0; 4], "checker_colors/1")?,
    ))
}

fn parse_pixels(engine: &Engine<'_>, value: Option<&Value>) -> Result<Vec<[f64; 4]>> {
    let values = value.and_then(Value::as_array).ok_or_else(|| {
        engine.error(
            ErrorCode::InvalidOperation,
            "pixels must be an array of RGBA colors",
            &operation_pointer(engine.operation_index, "pixels"),
        )
    })?;
    let mut result = Vec::new();
    for (row, value) in values.iter().enumerate() {
        if value.as_array().is_some_and(|components| {
            components.len() == 4 && components.iter().all(Value::is_number)
        }) {
            result.push(read_color(engine, Some(value), [0.0; 4], "pixels")?);
        } else if let Some(columns) = value.as_array() {
            for (column, color) in columns.iter().enumerate() {
                result.push(read_color(
                    engine,
                    Some(color),
                    [0.0; 4],
                    &format!("pixels/{row}/{column}"),
                )?);
            }
        } else {
            return Err(engine.error(
                ErrorCode::InvalidOperation,
                "pixels must contain RGBA arrays",
                &operation_pointer(engine.operation_index, &format!("pixels/{row}")),
            ));
        }
    }
    Ok(result)
}

fn read_dimension(engine: &Engine<'_>, operation: &Map<String, Value>, key: &str) -> Result<u32> {
    let value = operation.get(key).ok_or_else(|| {
        engine.error(
            ErrorCode::InvalidOperation,
            format!("missing required field `{key}`"),
            &operation_pointer(engine.operation_index, key),
        )
    })?;
    read_dimension_value(engine, value, key)
}

fn read_dimension_value(engine: &Engine<'_>, value: &Value, field: &str) -> Result<u32> {
    value
        .as_u64()
        .and_then(|number| u32::try_from(number).ok())
        .filter(|number| *number > 0)
        .ok_or_else(|| {
            engine.error(
                ErrorCode::InvalidOperation,
                "image dimensions must be positive u32 integers",
                &operation_pointer(engine.operation_index, field),
            )
        })
}

fn optional_dimension(
    engine: &Engine<'_>,
    operation: &Map<String, Value>,
    key: &str,
    default: u32,
) -> Result<u32> {
    operation
        .get(key)
        .map(|value| read_dimension_value(engine, value, key))
        .transpose()
        .map(|value| value.unwrap_or(default))
}

fn read_positive_integer(engine: &Engine<'_>, value: &Value, field: &str) -> Result<usize> {
    value
        .as_u64()
        .and_then(|number| usize::try_from(number).ok())
        .filter(|number| *number > 0)
        .ok_or_else(|| {
            engine.error(
                ErrorCode::InvalidOperation,
                "value must be a positive integer",
                &operation_pointer(engine.operation_index, field),
            )
        })
}
fn read_positive_u32(engine: &Engine<'_>, value: &Value, field: &str) -> Result<u32> {
    value
        .as_u64()
        .and_then(|number| u32::try_from(number).ok())
        .filter(|number| *number > 0)
        .ok_or_else(|| {
            engine.error(
                ErrorCode::InvalidOperation,
                "tile must be a positive u32 integer",
                &operation_pointer(engine.operation_index, field),
            )
        })
}

fn validate_dimensions(engine: &Engine<'_>, width: u32, height: u32) -> Result<()> {
    let count = u64::from(width).checked_mul(u64::from(height));
    if width == 0 || height == 0 || count.is_none_or(|count| count > crate::image::MAX_IMAGE_PIXELS)
    {
        return Err(engine.error(
            ErrorCode::LimitExceeded,
            "image dimensions exceed the supported pixel limit",
            &operation_pointer(engine.operation_index, "width"),
        ));
    }
    Ok(())
}

fn pixel_count(width: u32, height: u32) -> Result<usize> {
    u64::from(width)
        .checked_mul(u64::from(height))
        .and_then(|count| usize::try_from(count).ok())
        .ok_or_else(|| {
            PotError::new(
                ErrorCode::LimitExceeded,
                "image dimensions exceed platform limits",
            )
        })
}

fn stage_blob(engine: &mut Engine<'_>, bytes: Vec<u8>) -> String {
    let digest = hash::sha256(&bytes);
    engine.pending_assets.insert(digest.clone(), bytes);
    digest
}
