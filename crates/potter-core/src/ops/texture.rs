use serde_json::{Map, Value};

use crate::{
    error::{ErrorCode, Result},
    geom::legacy_texture,
};

use super::{ChangeKind, Engine, check_fields, operation_pointer, read_id};

pub(super) fn apply(
    engine: &mut Engine<'_>,
    name: &str,
    operation: &Map<String, Value>,
) -> Result<bool> {
    match name {
        "texture.create" => create(engine, operation),
        "texture.update" => update(engine, operation),
        _ => Err(engine.error(
            ErrorCode::InternalError,
            "invalid texture operation dispatch",
            &operation_pointer(engine.operation_index, "op"),
        )),
    }
}

/// `texture.create`: a Blender legacy texture as an inline `legacy_texture` record in `doc.resources`.
/// IMAGE textures reference a shared `doc.images` entry rather than embedding image pixels.
fn create(engine: &mut Engine<'_>, operation: &Map<String, Value>) -> Result<bool> {
    check_fields(
        engine,
        operation,
        &["op", "id", "texture_type", "image", "settings"],
        &["id", "texture_type"],
    )?;
    let id = read_id(engine, operation, "id")?;
    if engine.doc.resources.contains_key(&id) || engine.doc.images.contains_key(&id) {
        return Err(engine.error(
            ErrorCode::IdExists,
            format!("texture `{id}` already exists"),
            &operation_pointer(engine.operation_index, "id"),
        ));
    }
    let texture_type = operation
        .get("texture_type")
        .and_then(Value::as_str)
        .ok_or_else(|| {
            engine.error(
                ErrorCode::InvalidOperation,
                "texture_type must be a string",
                &operation_pointer(engine.operation_index, "texture_type"),
            )
        })?;
    let image = if texture_type == "IMAGE" {
        image_reference_field(engine, operation)?
    } else {
        if operation.contains_key("image") {
            return Err(engine.error(
                ErrorCode::InvalidOperation,
                "only IMAGE textures may reference an image",
                &operation_pointer(engine.operation_index, "image"),
            ));
        }
        None
    };
    let settings = settings_field(engine, operation)?;
    let record =
        legacy_texture::merged_record(&id.to_string(), texture_type, image.as_ref(), None, &settings)?;
    engine.doc.resources.insert(id.clone(), record);
    engine.mark("resources", &id, ChangeKind::Created);
    Ok(true)
}

/// `texture.update`: merges settings and, for IMAGE records, an optional shared image reference.
fn update(engine: &mut Engine<'_>, operation: &Map<String, Value>) -> Result<bool> {
    check_fields(
        engine,
        operation,
        &["op", "id", "image", "settings"],
        &["id"],
    )?;
    let id = read_id(engine, operation, "id")?;
    let existing = engine.doc.resources.get(&id).cloned().ok_or_else(|| {
        engine.error(
            ErrorCode::TargetNotFound,
            format!("texture `{id}` was not found"),
            &operation_pointer(engine.operation_index, "id"),
        )
    })?;
    if crate::model::migrate::inline_resource_kind(&existing)?
        != Some(legacy_texture::LEGACY_TEXTURE_KIND)
    {
        return Err(engine.error(
            ErrorCode::InvalidOperation,
            format!("resource `{id}` is not a legacy texture"),
            &operation_pointer(engine.operation_index, "id"),
        ));
    }
    let existing_texture = legacy_texture::parse_record(&id.to_string(), &existing)?;
    let (texture_type, existing_image) = match existing_texture {
        legacy_texture::LegacyTexture::Blend(_) => ("BLEND".to_owned(), None),
        legacy_texture::LegacyTexture::ImageRecord { image, .. } => {
            ("IMAGE".to_owned(), image)
        }
        legacy_texture::LegacyTexture::NativeValueNoise => {
            ("NATIVE_VALUE_NOISE".to_owned(), None)
        }
        legacy_texture::LegacyTexture::Procedural { texture_type, .. } => (texture_type, None),
    };
    let image = if texture_type == "IMAGE" {
        if operation.contains_key("image") {
            image_reference_field(engine, operation)?
        } else {
            existing_image
        }
    } else {
        if operation.contains_key("image") {
            return Err(engine.error(
                ErrorCode::InvalidOperation,
                "only IMAGE textures may reference an image",
                &operation_pointer(engine.operation_index, "image"),
            ));
        }
        None
    };
    let base = existing
        .get("settings")
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();
    let settings = settings_field(engine, operation)?;
    let record = legacy_texture::merged_record(
        &id.to_string(),
        &texture_type,
        image.as_ref(),
        Some(&base),
        &settings,
    )?;
    engine.doc.resources.insert(id.clone(), record);
    engine.mark("resources", &id, ChangeKind::Updated);
    Ok(true)
}

fn image_reference_field(
    engine: &Engine<'_>,
    operation: &Map<String, Value>,
) -> Result<Option<crate::model::Id>> {
    match operation.get("image") {
        Some(Value::Null) => Ok(None),
        Some(_) => image_id_field(engine, operation).map(Some),
        None => Err(engine.error(
            ErrorCode::InvalidOperation,
            "IMAGE textures require an image ID or null",
            &operation_pointer(engine.operation_index, "image"),
        )),
    }
}
fn image_id_field(engine: &Engine<'_>, operation: &Map<String, Value>) -> Result<crate::model::Id> {
    let image = read_id(engine, operation, "image")?;
    if !engine.doc.images.contains_key(&image) {
        return Err(engine.error(
            ErrorCode::TargetNotFound,
            format!("image `{image}` was not found"),
            &operation_pointer(engine.operation_index, "image"),
        ));
    }
    Ok(image)
}


fn settings_field(engine: &Engine<'_>, operation: &Map<String, Value>) -> Result<Map<String, Value>> {
    match operation.get("settings") {
        None => Ok(Map::new()),
        Some(Value::Object(settings)) => Ok(settings.clone()),
        Some(_) => Err(engine.error(
            ErrorCode::InvalidOperation,
            "texture settings must be an object",
            &operation_pointer(engine.operation_index, "settings"),
        )),
    }
}
