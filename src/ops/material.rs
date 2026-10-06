use serde_json::{Map, Value};

use crate::{
    error::{ErrorCode, Result},
    image::ImageInterpolation,
    model::{Id, Material, TextureRef},
};

use super::{
    ChangeKind, Engine, check_fields, operation_pointer, pointer_escape, read_bool, read_id,
    read_string, resolve_material_targets,
};

pub(super) fn apply(
    engine: &mut Engine<'_>,
    name: &str,
    operation: &Map<String, Value>,
) -> Result<bool> {
    match name {
        "material.create" => create(engine, operation),
        "material.update" => update(engine, operation),
        "material.delete" => delete(engine, operation),
        _ => Err(engine.error(
            ErrorCode::InternalError,
            "invalid material operation dispatch",
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
            "alpha_mode",
            "alpha_threshold",
            "base_color",
            "metallic",
            "roughness",
            "double_sided",
            "emission_color",
            "emission_strength",
            "transmission",
            "ior",
            "graph",
            "base_color_texture",
            "roughness_texture",
            "metallic_texture",
            "normal_texture",
            "displacement_method",
            "volume_density",
            "volume_color",
            "volume_anisotropy",
            "displacement_scale",
            "displacement_midlevel",
        ],
        &["id"],
    )?;
    let id = read_id(engine, operation, "id")?;
    if engine.doc.materials.contains_key(&id) {
        return Err(engine.error(
            ErrorCode::IdExists,
            format!("material ID `{id}` already exists"),
            &operation_pointer(engine.operation_index, "id"),
        ));
    }
    let mut material = Material {
        name: operation.get("name").map_or_else(
            || Ok(id.to_string()),
            |_| read_string(engine, operation, "name"),
        )?,
        base_color: [0.6, 0.6, 0.6, 1.0],
        metallic: 0.0,
        roughness: 0.8,
        emission_color: [0.0; 3],
        emission_strength: 0.0,
        transmission: 0.0,
        ior: 1.45,
        double_sided: false,
        ..Material::default()
    };
    if let Some(value) = operation.get("alpha_mode") {
        material.alpha_mode = parse_alpha_mode(
            engine,
            value,
            &operation_pointer(engine.operation_index, "alpha_mode"),
        )?;
    }
    if let Some(value) = operation.get("alpha_threshold") {
        material.alpha_threshold = read_unit_interval(engine, value, "alpha_threshold")?;
    }
    if let Some(value) = operation.get("displacement_method") {
        material.displacement_method = parse_displacement_method(
            engine,
            value,
            &operation_pointer(engine.operation_index, "displacement_method"),
        )?;
    }
    if let Some(value) = operation.get("volume_density") {
        material.volume_density = read_non_negative(
            engine,
            value,
            "volume_density",
            &operation_pointer(engine.operation_index, "volume_density"),
        )?;
    }
    if let Some(value) = operation.get("volume_color") {
        material.volume_color = read_emission_color(
            engine,
            value,
            &operation_pointer(engine.operation_index, "volume_color"),
        )?;
    }
    if let Some(value) = operation.get("volume_anisotropy") {
        material.volume_anisotropy = read_anisotropy(
            engine,
            value,
            &operation_pointer(engine.operation_index, "volume_anisotropy"),
        )?;
    }
    if let Some(value) = operation.get("displacement_scale") {
        material.displacement_scale = read_finite_number(
            engine,
            value,
            "displacement_scale",
            &operation_pointer(engine.operation_index, "displacement_scale"),
        )?;
    }
    if let Some(value) = operation.get("displacement_midlevel") {
        material.displacement_midlevel =
            read_unit_interval(engine, value, "displacement_midlevel")?;
    }
    if let Some(value) = operation.get("base_color") {
        material.base_color = read_color(
            engine,
            value,
            &operation_pointer(engine.operation_index, "base_color"),
        )?;
    }
    if let Some(value) = operation.get("metallic") {
        material.metallic = read_unit_interval(engine, value, "metallic")?;
    }
    if let Some(value) = operation.get("roughness") {
        material.roughness = read_unit_interval(engine, value, "roughness")?;
    }
    if operation.contains_key("double_sided") {
        material.double_sided = read_bool(engine, operation, "double_sided", false)?;
    }
    if let Some(value) = operation.get("emission_color") {
        material.emission_color = read_emission_color(
            engine,
            value,
            &operation_pointer(engine.operation_index, "emission_color"),
        )?;
    }
    if let Some(value) = operation.get("emission_strength") {
        material.emission_strength = read_non_negative(
            engine,
            value,
            "emission_strength",
            &operation_pointer(engine.operation_index, "emission_strength"),
        )?;
    }
    if let Some(value) = operation.get("transmission") {
        material.transmission = read_unit_interval(engine, value, "transmission")?;
    }
    if let Some(value) = operation.get("ior") {
        material.ior = read_positive(
            engine,
            value,
            "ior",
            &operation_pointer(engine.operation_index, "ior"),
        )?;
    }
    if let Some(graph) = operation.get("graph") {
        material.node_tree = Some(super::shader::resolve_graph(engine, graph)?);
    } else {
        super::shader::attach_default_graph(engine, &id, &mut material)?;
    }
    material.base_color_texture = operation
        .get("base_color_texture")
        .map(|value| parse_texture_ref(engine, value, "base_color_texture"))
        .transpose()?
        .flatten();
    material.roughness_texture = operation
        .get("roughness_texture")
        .map(|value| parse_texture_ref(engine, value, "roughness_texture"))
        .transpose()?
        .flatten();
    material.metallic_texture = operation
        .get("metallic_texture")
        .map(|value| parse_texture_ref(engine, value, "metallic_texture"))
        .transpose()?
        .flatten();
    material.normal_texture = operation
        .get("normal_texture")
        .map(|value| parse_texture_ref(engine, value, "normal_texture"))
        .transpose()?
        .flatten();
    engine.doc.materials.insert(id.clone(), material);
    engine.mark("materials", &id, ChangeKind::Created);
    Ok(true)
}

fn update(engine: &mut Engine<'_>, operation: &Map<String, Value>) -> Result<bool> {
    check_fields(
        engine,
        operation,
        &["op", "target", "set"],
        &["target", "set"],
    )?;
    let target = operation.get("target").ok_or_else(|| {
        engine.error(
            ErrorCode::InvalidOperation,
            "missing target",
            &operation_pointer(engine.operation_index, "target"),
        )
    })?;
    let targets = resolve_material_targets(engine, target, true)?;
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
    for field in set.keys() {
        if ![
            "name",
            "alpha_mode",
            "alpha_threshold",
            "base_color",
            "metallic",
            "roughness",
            "double_sided",
            "emission_color",
            "emission_strength",
            "transmission",
            "ior",
            "base_color_texture",
            "roughness_texture",
            "metallic_texture",
            "normal_texture",
            "displacement_method",
            "volume_density",
            "volume_color",
            "volume_anisotropy",
            "displacement_scale",
            "displacement_midlevel",
        ]
        .contains(&field.as_str())
        {
            return Err(engine.error(
                ErrorCode::InvalidOperation,
                format!("unknown material set field `{field}`"),
                &operation_pointer(
                    engine.operation_index,
                    &format!("set/{}", pointer_escape(field)),
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
    let base_color = set
        .get("base_color")
        .map(|value| {
            read_color(
                engine,
                value,
                &operation_pointer(engine.operation_index, "set/base_color"),
            )
        })
        .transpose()?;
    let metallic = set
        .get("metallic")
        .map(|value| read_unit_interval(engine, value, "set/metallic"))
        .transpose()?;
    let roughness = set
        .get("roughness")
        .map(|value| read_unit_interval(engine, value, "set/roughness"))
        .transpose()?;
    let emission_color = set
        .get("emission_color")
        .map(|value| {
            read_emission_color(
                engine,
                value,
                &operation_pointer(engine.operation_index, "set/emission_color"),
            )
        })
        .transpose()?;
    let emission_strength = set
        .get("emission_strength")
        .map(|value| {
            read_non_negative(
                engine,
                value,
                "emission_strength",
                &operation_pointer(engine.operation_index, "set/emission_strength"),
            )
        })
        .transpose()?;
    let transmission = set
        .get("transmission")
        .map(|value| read_unit_interval(engine, value, "set/transmission"))
        .transpose()?;
    let ior = set
        .get("ior")
        .map(|value| {
            read_positive(
                engine,
                value,
                "ior",
                &operation_pointer(engine.operation_index, "set/ior"),
            )
        })
        .transpose()?;
    let double_sided = set
        .get("double_sided")
        .map(|value| {
            value.as_bool().ok_or_else(|| {
                engine.error(
                    ErrorCode::InvalidOperation,
                    "double_sided must be a boolean",
                    &operation_pointer(engine.operation_index, "set/double_sided"),
                )
            })
        })
        .transpose()?;
    let base_color_texture = set
        .get("base_color_texture")
        .map(|value| parse_texture_ref(engine, value, "set/base_color_texture"))
        .transpose()?;
    let roughness_texture = set
        .get("roughness_texture")
        .map(|value| parse_texture_ref(engine, value, "set/roughness_texture"))
        .transpose()?;
    let metallic_texture = set
        .get("metallic_texture")
        .map(|value| parse_texture_ref(engine, value, "set/metallic_texture"))
        .transpose()?;
    let normal_texture = set
        .get("normal_texture")
        .map(|value| parse_texture_ref(engine, value, "set/normal_texture"))
        .transpose()?;
    let alpha_mode = set
        .get("alpha_mode")
        .map(|value| {
            parse_alpha_mode(
                engine,
                value,
                &operation_pointer(engine.operation_index, "set/alpha_mode"),
            )
        })
        .transpose()?;
    let alpha_threshold = set
        .get("alpha_threshold")
        .map(|value| read_unit_interval(engine, value, "set/alpha_threshold"))
        .transpose()?;
    let displacement_method = set
        .get("displacement_method")
        .map(|value| {
            parse_displacement_method(
                engine,
                value,
                &operation_pointer(engine.operation_index, "set/displacement_method"),
            )
        })
        .transpose()?;
    let volume_density = set
        .get("volume_density")
        .map(|value| {
            read_non_negative(
                engine,
                value,
                "volume_density",
                &operation_pointer(engine.operation_index, "set/volume_density"),
            )
        })
        .transpose()?;
    let volume_color = set
        .get("volume_color")
        .map(|value| {
            read_emission_color(
                engine,
                value,
                &operation_pointer(engine.operation_index, "set/volume_color"),
            )
        })
        .transpose()?;
    let volume_anisotropy = set
        .get("volume_anisotropy")
        .map(|value| {
            read_anisotropy(
                engine,
                value,
                &operation_pointer(engine.operation_index, "set/volume_anisotropy"),
            )
        })
        .transpose()?;
    let displacement_scale = set
        .get("displacement_scale")
        .map(|value| {
            read_finite_number(
                engine,
                value,
                "displacement_scale",
                &operation_pointer(engine.operation_index, "set/displacement_scale"),
            )
        })
        .transpose()?;
    let displacement_midlevel = set
        .get("displacement_midlevel")
        .map(|value| read_unit_interval(engine, value, "set/displacement_midlevel"))
        .transpose()?;
    let mut changed = false;
    for id in targets {
        let item_changed = if let Some(material) = engine.doc.materials.get_mut(&id) {
            let mut item_changed = false;
            if let Some(name) = &name
                && material.name != *name
            {
                material.name.clone_from(name);
                item_changed = true;
            }
            if let Some(value) = &alpha_mode
                && material.alpha_mode != *value
            {
                material.alpha_mode.clone_from(value);
                item_changed = true;
            }
            if let Some(value) = alpha_threshold
                && !crate::float::equal_f64(material.alpha_threshold, value)
            {
                material.alpha_threshold = value;
                item_changed = true;
            }
            if let Some(color) = base_color
                && !crate::float::equal_f64_array(&material.base_color, &color)
            {
                material.base_color = color;
                item_changed = true;
            }
            if let Some(value) = metallic
                && !crate::float::equal_f64(material.metallic, value)
            {
                material.metallic = value;
                item_changed = true;
            }
            if let Some(value) = roughness
                && !crate::float::equal_f64(material.roughness, value)
            {
                material.roughness = value;
                item_changed = true;
            }
            if let Some(value) = emission_color
                && !crate::float::equal_f64_array(&material.emission_color, &value)
            {
                material.emission_color = value;
                item_changed = true;
            }
            if let Some(value) = emission_strength
                && !crate::float::equal_f64(material.emission_strength, value)
            {
                material.emission_strength = value;
                item_changed = true;
            }
            if let Some(value) = transmission
                && !crate::float::equal_f64(material.transmission, value)
            {
                material.transmission = value;
                item_changed = true;
            }
            if let Some(value) = ior
                && !crate::float::equal_f64(material.ior, value)
            {
                material.ior = value;
                item_changed = true;
            }
            if let Some(value) = double_sided
                && material.double_sided != value
            {
                material.double_sided = value;
                item_changed = true;
            }
            if let Some(value) = &base_color_texture
                && material.base_color_texture != *value
            {
                material.base_color_texture.clone_from(value);
                item_changed = true;
            }
            if let Some(value) = &roughness_texture
                && material.roughness_texture != *value
            {
                material.roughness_texture.clone_from(value);
                item_changed = true;
            }
            if let Some(value) = &metallic_texture
                && material.metallic_texture != *value
            {
                material.metallic_texture.clone_from(value);
                item_changed = true;
            }
            if let Some(value) = &normal_texture
                && material.normal_texture != *value
            {
                material.normal_texture.clone_from(value);
                item_changed = true;
            }
            if let Some(value) = &displacement_method
                && material.displacement_method != *value
            {
                material.displacement_method.clone_from(value);
                item_changed = true;
            }
            if let Some(value) = volume_density
                && !crate::float::equal_f64(material.volume_density, value)
            {
                material.volume_density = value;
                item_changed = true;
            }
            if let Some(value) = volume_color
                && !crate::float::equal_f64_array(&material.volume_color, &value)
            {
                material.volume_color = value;
                item_changed = true;
            }
            if let Some(value) = volume_anisotropy
                && !crate::float::equal_f64(material.volume_anisotropy, value)
            {
                material.volume_anisotropy = value;
                item_changed = true;
            }
            if let Some(value) = displacement_scale
                && !crate::float::equal_f64(material.displacement_scale, value)
            {
                material.displacement_scale = value;
                item_changed = true;
            }
            if let Some(value) = displacement_midlevel
                && !crate::float::equal_f64(material.displacement_midlevel, value)
            {
                material.displacement_midlevel = value;
                item_changed = true;
            }
            item_changed
        } else {
            false
        };
        if item_changed {
            engine.mark("materials", &id, ChangeKind::Updated);
            changed = true;
        }
    }
    Ok(changed)
}

fn delete(engine: &mut Engine<'_>, operation: &Map<String, Value>) -> Result<bool> {
    check_fields(
        engine,
        operation,
        &["op", "target", "reassign"],
        &["target"],
    )?;
    let targets = resolve_material_targets(
        engine,
        operation.get("target").ok_or_else(|| {
            engine.error(
                ErrorCode::InvalidOperation,
                "missing target",
                &operation_pointer(engine.operation_index, "target"),
            )
        })?,
        true,
    )?;
    let pointer = operation_pointer(engine.operation_index, "reassign");
    let reassign = operation.get("reassign").map(|value| {
        let policy = value.as_str().ok_or_else(|| {
            engine.error(ErrorCode::InvalidOperation, "reassign must be `default` or a material ID", &pointer)
        })?;
        if policy == "default" {
            Ok(None)
        } else {
            let id = Id::new(policy).map_err(|error| {
                engine.error(ErrorCode::InvalidOperation, error.message, &pointer)
            })?;
            if !engine.doc.materials.contains_key(&id) {
                return Err(engine.error(ErrorCode::TargetNotFound, format!("reassignment material `{id}` was not found"), &pointer));
            }
            if targets.contains(&id) {
                return Err(engine.error(ErrorCode::InvalidOperation, "cannot reassign a deleted material to itself or another material being deleted", &pointer));
            }
            Ok(Some(id))
        }
    }).transpose()?;
    for id in &targets {
        let is_referenced = engine
            .doc
            .nodes
            .values()
            .any(|node| node.materials.contains(id));
        if is_referenced && reassign.is_none() {
            return Err(engine.error(
                ErrorCode::InvalidOperation,
                format!(
                    "material `{id}` is assigned to nodes; provide an explicit reassign policy"
                ),
                &pointer,
            ));
        }
    }
    let default_graphs = targets
        .iter()
        .filter_map(|id| engine.doc.materials.get(id)?.node_tree.as_ref())
        .filter(|graph_id| {
            engine
                .doc
                .node_groups
                .get(*graph_id)
                .is_some_and(crate::graph::is_generated_simple_material_graph)
        })
        .cloned()
        .collect::<std::collections::BTreeSet<_>>();
    let mut changed_nodes = std::collections::BTreeSet::new();
    for (node_id, node) in &mut engine.doc.nodes {
        let mut changed = false;
        match &reassign {
            Some(Some(replacement)) => {
                for material in &mut node.materials {
                    if targets.contains(material) {
                        *material = replacement.clone();
                        changed = true;
                    }
                }
            }
            Some(None) => {
                let old_len = node.materials.len();
                node.materials
                    .retain(|material| !targets.contains(material));
                changed = node.materials.len() != old_len;
            }
            None => {}
        }
        if changed {
            changed_nodes.insert(node_id.clone());
        }
    }
    for id in targets {
        if engine.doc.materials.remove(&id).is_some() {
            engine.mark("materials", &id, ChangeKind::Deleted);
        }
    }
    for graph_id in default_graphs {
        let still_referenced = engine
            .doc
            .materials
            .values()
            .any(|material| material.node_tree.as_ref() == Some(&graph_id));
        if !still_referenced && engine.doc.node_groups.remove(&graph_id).is_some() {
            engine.mark("node_groups", &graph_id, ChangeKind::Deleted);
        }
    }
    for node_id in changed_nodes {
        engine.mark("nodes", &node_id, ChangeKind::Updated);
    }
    Ok(true)
}

fn parse_texture_ref(
    engine: &Engine<'_>,
    value: &Value,
    pointer: &str,
) -> Result<Option<TextureRef>> {
    if value.is_null() {
        return Ok(None);
    }
    let object = value.as_object().ok_or_else(|| {
        engine.error(
            ErrorCode::InvalidOperation,
            "texture reference must be an object or null",
            &operation_pointer(engine.operation_index, pointer),
        )
    })?;
    for key in object.keys() {
        if !["image", "uv_map", "interpolation"].contains(&key.as_str()) {
            return Err(engine.error(
                ErrorCode::InvalidOperation,
                format!("unknown texture reference field `{key}`"),
                &operation_pointer(
                    engine.operation_index,
                    &format!("{pointer}/{}", pointer_escape(key)),
                ),
            ));
        }
    }
    let image_name = object.get("image").and_then(Value::as_str).ok_or_else(|| {
        engine.error(
            ErrorCode::InvalidOperation,
            "texture reference requires an image ID",
            &operation_pointer(engine.operation_index, &format!("{pointer}/image")),
        )
    })?;
    let image_id = super::parse_id(
        engine,
        image_name,
        &operation_pointer(engine.operation_index, &format!("{pointer}/image")),
    )?;
    if !engine.doc.images.contains_key(&image_id) {
        return Err(engine.error(
            ErrorCode::TargetNotFound,
            format!("texture image `{image_id}` was not found"),
            &operation_pointer(engine.operation_index, &format!("{pointer}/image")),
        ));
    }
    let uv_map = match object.get("uv_map") {
        None | Some(Value::Null) => None,
        Some(Value::String(uv_map)) => Some(uv_map.clone()),
        Some(_) => {
            return Err(engine.error(
                ErrorCode::InvalidOperation,
                "uv_map must be a string or null",
                &operation_pointer(engine.operation_index, &format!("{pointer}/uv_map")),
            ));
        }
    };
    let interpolation = match object.get("interpolation") {
        None | Some(Value::Null) => ImageInterpolation::Linear,
        Some(Value::String(value)) if value == "linear" => ImageInterpolation::Linear,
        Some(Value::String(value)) if value == "closest" => ImageInterpolation::Closest,
        _ => {
            return Err(engine.error(
                ErrorCode::InvalidOperation,
                "texture interpolation must be linear or closest",
                &operation_pointer(engine.operation_index, &format!("{pointer}/interpolation")),
            ));
        }
    };
    Ok(Some(TextureRef {
        image: image_id,
        uv_map,
        interpolation,
    }))
}

fn read_color(engine: &Engine<'_>, value: &Value, pointer: &str) -> Result<[f64; 4]> {
    let Some(values) = value.as_array() else {
        return Err(engine.error(
            ErrorCode::InvalidOperation,
            "base_color must be a four-element array",
            pointer,
        ));
    };
    if values.len() != 4 {
        return Err(engine.error(
            ErrorCode::InvalidOperation,
            "base_color must be a four-element array",
            pointer,
        ));
    }
    let mut color = [0.0; 4];
    for (index, component) in values.iter().enumerate() {
        let Some(component) = component.as_f64() else {
            return Err(engine.error(
                ErrorCode::InvalidOperation,
                "base_color components must be finite numbers",
                &format!("{pointer}/{index}"),
            ));
        };
        if !(0.0..=1.0).contains(&component) {
            return Err(engine.error(
                ErrorCode::InvalidOperation,
                "base_color components must be in [0, 1]",
                &format!("{pointer}/{index}"),
            ));
        }
        color[index] = component;
    }
    Ok(color)
}

fn read_emission_color(engine: &Engine<'_>, value: &Value, pointer: &str) -> Result<[f64; 3]> {
    let values = value.as_array().ok_or_else(|| {
        engine.error(
            ErrorCode::InvalidOperation,
            "emission_color must be a three-element array",
            pointer,
        )
    })?;
    if values.len() != 3 {
        return Err(engine.error(
            ErrorCode::InvalidOperation,
            "emission_color must be a three-element array",
            pointer,
        ));
    }
    let mut color = [0.0; 3];
    for (index, component) in values.iter().enumerate() {
        let component = component
            .as_f64()
            .filter(|number| number.is_finite())
            .ok_or_else(|| {
                engine.error(
                    ErrorCode::InvalidOperation,
                    "emission_color components must be finite numbers",
                    &format!("{pointer}/{index}"),
                )
            })?;
        if component < 0.0 {
            return Err(engine.error(
                ErrorCode::InvalidOperation,
                "emission_color components must be non-negative",
                &format!("{pointer}/{index}"),
            ));
        }
        color[index] = component;
    }
    Ok(color)
}

fn read_finite_number(
    engine: &Engine<'_>,
    value: &Value,
    field: &str,
    pointer: &str,
) -> Result<f64> {
    value
        .as_f64()
        .filter(|number| number.is_finite())
        .ok_or_else(|| {
            engine.error(
                ErrorCode::InvalidOperation,
                format!("{field} must be a finite number"),
                pointer,
            )
        })
}

fn read_anisotropy(engine: &Engine<'_>, value: &Value, pointer: &str) -> Result<f64> {
    let anisotropy = read_finite_number(engine, value, "volume_anisotropy", pointer)?;
    if !(-0.999..=0.999).contains(&anisotropy) {
        return Err(engine.error(
            ErrorCode::InvalidOperation,
            "volume_anisotropy must be in -0.999..=0.999",
            pointer,
        ));
    }
    Ok(anisotropy)
}
fn read_non_negative(
    engine: &Engine<'_>,
    value: &Value,
    field: &str,
    pointer: &str,
) -> Result<f64> {
    let number = read_finite_number(engine, value, field, pointer)?;
    if number < 0.0 {
        return Err(engine.error(
            ErrorCode::InvalidOperation,
            format!("{field} must be non-negative"),
            pointer,
        ));
    }
    Ok(number)
}

fn read_positive(engine: &Engine<'_>, value: &Value, field: &str, pointer: &str) -> Result<f64> {
    let number = read_finite_number(engine, value, field, pointer)?;
    if number <= 0.0 {
        return Err(engine.error(
            ErrorCode::InvalidOperation,
            format!("{field} must be positive"),
            pointer,
        ));
    }
    Ok(number)
}

fn parse_displacement_method(engine: &Engine<'_>, value: &Value, pointer: &str) -> Result<String> {
    match value.as_str() {
        Some("bump" | "displacement") => value.as_str().map(str::to_owned).ok_or_else(|| {
            engine.error(
                ErrorCode::InvalidOperation,
                "displacement_method must be bump or displacement",
                pointer,
            )
        }),
        _ => Err(engine.error(
            ErrorCode::InvalidOperation,
            "displacement_method must be bump or displacement",
            pointer,
        )),
    }
}

fn parse_alpha_mode(engine: &Engine<'_>, value: &Value, pointer: &str) -> Result<String> {
    match value.as_str() {
        Some("opaque" | "blend" | "clip") => value.as_str().map(str::to_owned).ok_or_else(|| {
            engine.error(
                ErrorCode::InvalidOperation,
                "alpha_mode must be opaque, blend, or clip",
                pointer,
            )
        }),
        _ => Err(engine.error(
            ErrorCode::InvalidOperation,
            "alpha_mode must be opaque, blend, or clip",
            pointer,
        )),
    }
}
fn read_unit_interval(engine: &Engine<'_>, value: &Value, field: &str) -> Result<f64> {
    let Some(number) = value.as_f64() else {
        return Err(engine.error(
            ErrorCode::InvalidOperation,
            format!("{field} must be a finite number"),
            &operation_pointer(engine.operation_index, field),
        ));
    };
    if !number.is_finite() || !(0.0..=1.0).contains(&number) {
        return Err(engine.error(
            ErrorCode::InvalidOperation,
            format!("{field} must be in [0, 1]"),
            &operation_pointer(engine.operation_index, field),
        ));
    }
    Ok(number)
}
