use serde_json::{Map, Value, json};

use crate::{
    error::{ErrorCode, PotError, Result},
    geom::{
        Mesh, ORIGINDEX_ATTRIBUTE, ORIGINDEX_NONE, attribute_keys, clnor,
        color_layers::{self, ColorDomain, ColorLayer, ColorStore},
        uv_layers::{self, UvLayer, UvStore},
    },
};

/// Transfers mesh data for Blender's Dual Mesh node. Every index slice is in output storage order.
pub(super) fn transfer(
    source: &Mesh,
    output: &mut Mesh,
    point_face_sources: &[Option<usize>],
    face_point_sources: &[usize],
    edge_sources: &[Option<usize>],
    corner_sources: &[usize],
) -> Result<()> {
    let source_corner_count = corner_count(source)?;
    let output_corner_count = corner_count(output)?;
    if point_face_sources.len() != output.vertices.len()
        || face_point_sources.len() != output.faces.len()
        || edge_sources.len() != output.edges.len()
        || corner_sources.len() != output_corner_count
        || point_face_sources
            .iter()
            .flatten()
            .any(|&index| index >= source.faces.len())
        || face_point_sources
            .iter()
            .any(|&index| index >= source.vertices.len())
        || edge_sources
            .iter()
            .flatten()
            .any(|&index| index >= source.edges.len())
        || corner_sources
            .iter()
            .any(|&index| index >= source_corner_count)
    {
        return Err(invalid(
            "Dual Mesh attribute provenance does not match its meshes",
        ));
    }
    color_layers::validate(source)?;
    uv_layers::validate(source)?;
    clnor::validate(source)?;

    let color_store = color_layers::read(source);
    let needs_corner_locations = uv_layers::has_uv_layers(source)
        || source.attributes.values().any(|value| {
            value
                .get("domain")
                .and_then(Value::as_str)
                .is_some_and(|domain| domain.eq_ignore_ascii_case("CORNER"))
                && value.get("values").and_then(Value::as_object).is_some()
        });
    let source_corners = if needs_corner_locations {
        corner_locations(source)?
    } else {
        Vec::new()
    };
    let mut attributes = Map::new();
    for (key, value) in &source.attributes {
        if matches!(
            key.as_str(),
            "position"
                | ".edge_verts"
                | ".corner_vert"
                | ".corner_edge"
                | "sharp_face"
                | "color_layers"
                | "uv_layers"
                | "custom_normal"
                | "vertex_groups"
        ) || color_store
            .layers
            .iter()
            .any(|layer| attribute_keys::storage_key(&layer.name).as_ref() == key)
        {
            continue;
        }
        let remapped = match key.as_str() {
            "blender_attributes" => Some(remap_blender_attributes(
                source,
                value,
                point_face_sources,
                face_point_sources,
                edge_sources,
                corner_sources,
                source_corner_count,
            )?),
            "blender_edge_flags" => Some(remap_edge_flags(value, source, edge_sources)?),
            "blender_polygon_smooth" => Some(remap_polygon_smooth(value, source, output)?),
            "shade_smooth" => Some(remap_shade_smooth(value, source, output)?),
            ORIGINDEX_ATTRIBUTE => Some(remap_origindex(value, source, output)?),
            "blender_skin_vertices" | "skin_radii" | "skin_roots" | "multires_displacement" => {
                // These are modifier/control data, not attributes exposed by Blender's mesh accessor.
                None
            }
            _ if value.get("domain").and_then(Value::as_str).is_some() => {
                Some(remap_named_attribute(
                    key,
                    value,
                    source,
                    output,
                    point_face_sources,
                    face_point_sources,
                    edge_sources,
                    corner_sources,
                    &source_corners,
                    source_corner_count,
                )?)
            }
            _ => Some(value.clone()),
        };
        if let Some(remapped) = remapped {
            attributes.insert(key.clone(), remapped);
        }
    }

    remap_vertex_groups(source, face_point_sources, &mut attributes)?;
    output.attributes = attributes;

    let output_colors = remap_colors(
        &color_store,
        output,
        point_face_sources,
        face_point_sources,
        corner_sources,
    )?;
    color_layers::write(output, &output_colors);
    remap_uv_layers(source, output, corner_sources, &source_corners)?;

    if source.attributes.contains_key("custom_normal") {
        let normal_sources = corner_sources.iter().copied().map(Some).collect::<Vec<_>>();
        clnor::transfer_custom_normals(
            source,
            output,
            &normal_sources,
            clnor::NormalCodeOptions::default(),
        )?;
    }
    Ok(())
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Domain {
    Point,
    Edge,
    Face,
    Corner,
}

#[derive(Clone, Copy)]
enum Indices<'a> {
    Optional(&'a [Option<usize>]),
    Exact(&'a [usize]),
}

impl Indices<'_> {
    fn get(self, index: usize) -> Option<usize> {
        match self {
            Self::Optional(indices) => indices.get(index).copied().flatten(),
            Self::Exact(indices) => indices.get(index).copied(),
        }
    }

    fn len(self) -> usize {
        match self {
            Self::Optional(indices) => indices.len(),
            Self::Exact(indices) => indices.len(),
        }
    }
}

fn domain_mapping<'a>(
    domain: Domain,
    point_face_sources: &'a [Option<usize>],
    face_point_sources: &'a [usize],
    edge_sources: &'a [Option<usize>],
    corner_sources: &'a [usize],
) -> (&'static str, Indices<'a>) {
    match domain {
        Domain::Point => ("FACE", Indices::Exact(face_point_sources)),
        Domain::Face => ("POINT", Indices::Optional(point_face_sources)),
        Domain::Edge => ("EDGE", Indices::Optional(edge_sources)),
        Domain::Corner => ("CORNER", Indices::Exact(corner_sources)),
    }
}

fn parse_domain(domain: &str, attribute: &str, data_type: Option<&str>) -> Result<Domain> {
    match domain.to_ascii_uppercase().as_str() {
        "POINT" => Ok(Domain::Point),
        "EDGE" => Ok(Domain::Edge),
        "FACE" => Ok(Domain::Face),
        "CORNER" => Ok(Domain::Corner),
        _ => Err(PotError::with_details(
            ErrorCode::UnsupportedFeature,
            format!("Dual Mesh cannot transfer `{attribute}` from domain `{domain}`"),
            json!({
                "feature_id": "graph.node.dual_mesh.attribute_domain",
                "attribute": attribute,
                "domain": domain,
                "data_type": data_type
            }),
        )),
    }
}

fn target_domain_name(source: &str, target: &str) -> String {
    if source
        .chars()
        .all(|character| !character.is_ascii_lowercase())
    {
        target.to_owned()
    } else if source
        .chars()
        .all(|character| !character.is_ascii_uppercase())
    {
        target.to_ascii_lowercase()
    } else {
        target.to_owned()
    }
}

fn source_count(source: &Mesh, domain: Domain, corner_count: usize) -> usize {
    match domain {
        Domain::Point => source.vertices.len(),
        Domain::Edge => source.edges.len(),
        Domain::Face => source.faces.len(),
        Domain::Corner => corner_count,
    }
}

fn output_element_key(mesh: &Mesh, domain: Domain, index: usize) -> Result<String> {
    match domain {
        Domain::Point => mesh
            .vertices
            .get(index)
            .map(|vertex| format!("v{}", vertex.id)),
        Domain::Edge => mesh.edges.get(index).map(|edge| format!("e{}", edge.id)),
        Domain::Face => mesh.faces.get(index).map(|face| format!("f{}", face.id)),
        Domain::Corner => None,
    }
    .ok_or_else(|| invalid("Dual Mesh named-attribute output index is out of range"))
}

fn source_element_key(mesh: &Mesh, domain: Domain, index: usize) -> Result<String> {
    match domain {
        Domain::Point => mesh
            .vertices
            .get(index)
            .map(|vertex| format!("v{}", vertex.id)),
        Domain::Edge => mesh.edges.get(index).map(|edge| format!("e{}", edge.id)),
        Domain::Face => mesh.faces.get(index).map(|face| format!("f{}", face.id)),
        Domain::Corner => None,
    }
    .ok_or_else(|| invalid("Dual Mesh named-attribute source index is out of range"))
}

fn named_data_type<'a>(key: &str, attribute: &'a Value) -> Option<&'a str> {
    attribute
        .get("type")
        .or_else(|| attribute.get("data_type"))
        .and_then(Value::as_str)
        .or(match key {
            "sharp" | "seam" | "shade_smooth" => Some("BOOLEAN"),
            "crease_edge" | "bevel_weight_edge" => Some("FLOAT"),
            _ => None,
        })
}

fn default_attribute_value(data_type: Option<&str>) -> Option<Value> {
    let kind = data_type?.to_ascii_uppercase();
    Some(match kind.as_str() {
        "BOOL" | "BOOLEAN" => json!(false),
        "INT" | "INT8" | "INT16" | "INT32" => json!(0),
        "FLOAT" => json!(0.0),
        "FLOAT2" => json!([0.0, 0.0]),
        "INT16_2D" | "INT32_2D" => json!([0, 0]),
        "FLOAT_VECTOR" => json!([0.0, 0.0, 0.0]),
        "FLOAT4" | "FLOAT_COLOR" | "BYTE_COLOR" | "COLOR" => json!([0.0, 0.0, 0.0, 0.0]),
        "QUATERNION" => json!([0.0, 0.0, 0.0, 1.0]),
        "FLOAT4X4" => json!([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ]),
        "STRING" => json!(""),
        _ => return None,
    })
}

fn missing_value(attribute: &str, data_type: Option<&str>) -> PotError {
    PotError::with_details(
        ErrorCode::UnsupportedFeature,
        format!(
            "Dual Mesh cannot initialize a missing `{attribute}` value of type `{}`",
            data_type.unwrap_or("unknown")
        ),
        json!({"attribute": attribute, "data_type": data_type}),
    )
}

fn source_value<'a>(
    values: &'a Map<String, Value>,
    source: &Mesh,
    domain: Domain,
    index: usize,
) -> Result<Option<&'a Value>> {
    let key = source_element_key(source, domain, index)?;
    Ok(values.get(&key))
}

fn remap_named_attribute(
    key: &str,
    attribute: &Value,
    source: &Mesh,
    output: &Mesh,
    point_face_sources: &[Option<usize>],
    face_point_sources: &[usize],
    edge_sources: &[Option<usize>],
    corner_sources: &[usize],
    source_corners: &[(usize, usize)],
    source_corner_count: usize,
) -> Result<Value> {
    let source_name = attribute
        .get("domain")
        .and_then(Value::as_str)
        .ok_or_else(|| invalid("Dual Mesh named attribute has no domain"))?;
    let data_type = named_data_type(key, attribute);
    let source_domain = parse_domain(source_name, key, data_type)?;
    let crease_vert_on_dual_faces = key == "crease_vert" && source_domain == Domain::Point;
    let (target_name, indices, target_domain) = if crease_vert_on_dual_faces {
        (source_name, Indices::Exact(face_point_sources), Domain::Face)
    } else {
        let (target_name, indices) = domain_mapping(
            source_domain,
            point_face_sources,
            face_point_sources,
            edge_sources,
            corner_sources,
        );
        let target_domain = match source_domain {
            Domain::Point => Domain::Face,
            Domain::Face => Domain::Point,
            Domain::Edge => Domain::Edge,
            Domain::Corner => Domain::Corner,
        };
        (target_name, indices, target_domain)
    };
    let type_name = data_type;
    let mut result = attribute
        .as_object()
        .cloned()
        .ok_or_else(|| invalid("Dual Mesh named attribute must be an object"))?;
    result.insert(
        "domain".to_owned(),
        json!(target_domain_name(source_name, target_name)),
    );

    let mut had_payload = false;
    if let Some(data) = attribute.get("data") {
        let values = data
            .as_array()
            .ok_or_else(|| invalid("Dual Mesh named attribute data must be an array"))?;
        had_payload = true;
        result.insert(
            "data".to_owned(),
            Value::Array(gather_array(
                values,
                indices,
                source_count(source, source_domain, source_corner_count),
                key,
                type_name,
            )?),
        );
    }
    if let Some(values) = attribute.get("values") {
        match values {
            Value::Array(values) => {
                had_payload = true;
                result.insert(
                    "values".to_owned(),
                    Value::Array(gather_array(
                        values,
                        indices,
                        source_count(source, source_domain, source_corner_count),
                        key,
                        type_name,
                    )?),
                );
            }
            Value::Object(values) => {
                had_payload = true;
                let remapped = remap_keyed_values(
                    key,
                    values,
                    source,
                    output,
                    source_domain,
                    target_domain,
                    indices,
                    type_name,
                    source_corners,
                )?;
                if crease_vert_on_dual_faces {
                    let face_values = output
                        .faces
                        .iter()
                        .map(|face| {
                            remapped
                                .get(&format!("f{}", face.id))
                                .cloned()
                                .ok_or_else(|| invalid("Dual Mesh face crease value is missing"))
                        })
                        .collect::<Result<Vec<_>>>()?;
                    result.insert("values".to_owned(), Value::Array(face_values));
                } else {
                    result.insert("values".to_owned(), Value::Object(remapped));
                }
            }
            _ => {
                return Err(invalid(
                    "Dual Mesh named attribute values must be an array or object",
                ));
            }
        }
    }
    if !had_payload {
        return Err(invalid(
            "Dual Mesh named attribute has no transferable values",
        ));
    }
    Ok(Value::Object(result))
}

fn gather_array(
    values: &[Value],
    indices: Indices<'_>,
    expected_source_count: usize,
    attribute: &str,
    data_type: Option<&str>,
) -> Result<Vec<Value>> {
    if values.len() != expected_source_count {
        return Err(invalid(format!(
            "Dual Mesh `{attribute}` value count {} does not match its source domain {expected_source_count}",
            values.len()
        )));
    }
    (0..indices.len())
        .map(|output_index| match indices.get(output_index) {
            Some(source_index) => values.get(source_index).cloned().ok_or_else(|| {
                invalid(format!(
                    "Dual Mesh `{attribute}` source index is out of range"
                ))
            }),
            None => default_attribute_value(data_type)
                .ok_or_else(|| missing_value(attribute, data_type)),
        })
        .collect()
}

fn remap_keyed_values(
    attribute: &str,
    values: &Map<String, Value>,
    source: &Mesh,
    output: &Mesh,
    source_domain: Domain,
    target_domain: Domain,
    indices: Indices<'_>,
    data_type: Option<&str>,
    source_corners: &[(usize, usize)],
) -> Result<Map<String, Value>> {
    let mut result = Map::new();
    if matches!(source_domain, Domain::Corner) {
        let mut output_corner = 0;
        for face in &output.faces {
            let mut row = Vec::with_capacity(face.vertices.len());
            for _ in &face.vertices {
                let source_corner = indices
                    .get(output_corner)
                    .ok_or_else(|| invalid("Dual Mesh corner provenance is missing"))?;
                let (source_face, local_corner) = *source_corners
                    .get(source_corner)
                    .ok_or_else(|| invalid("Dual Mesh source corner is out of range"))?;
                let source_face = source
                    .faces
                    .get(source_face)
                    .ok_or_else(|| invalid("Dual Mesh source face is out of range"))?;
                let source_row = values
                    .get(&format!("f{}", source_face.id))
                    .and_then(Value::as_array);
                let value = source_row.and_then(|row| row.get(local_corner));
                row.push(match value {
                    Some(value) => value.clone(),
                    None => default_attribute_value(data_type)
                        .ok_or_else(|| missing_value(attribute, data_type))?,
                });
                output_corner += 1;
            }
            result.insert(format!("f{}", face.id), Value::Array(row));
        }
        return Ok(result);
    }
    for output_index in 0..indices.len() {
        let output_key = output_element_key(output, target_domain, output_index)?;
        let value = indices
            .get(output_index)
            .map(|source_index| source_value(values, source, source_domain, source_index))
            .transpose()?
            .flatten();
        result.insert(
            output_key,
            match value {
                Some(value) => value.clone(),
                None => default_attribute_value(data_type)
                    .ok_or_else(|| missing_value(attribute, data_type))?,
            },
        );
    }
    Ok(result)
}

fn remap_blender_attributes(
    source: &Mesh,
    attributes: &Value,
    point_face_sources: &[Option<usize>],
    face_point_sources: &[usize],
    edge_sources: &[Option<usize>],
    corner_sources: &[usize],
    source_corner_count: usize,
) -> Result<Value> {
    let entries = attributes
        .as_array()
        .ok_or_else(|| invalid("Dual Mesh Blender attributes must be an array"))?;
    entries
        .iter()
        .map(|entry| {
            let name = entry
                .get("name")
                .and_then(Value::as_str)
                .ok_or_else(|| invalid("Dual Mesh Blender attribute has no name"))?;
            if is_builtin_excluded(name) {
                return Ok(None);
            }
            let domain_name = entry
                .get("domain")
                .and_then(Value::as_str)
                .ok_or_else(|| invalid("Dual Mesh Blender attribute has no domain"))?;
            let data_type = entry.get("data_type").and_then(Value::as_str);
            let domain = parse_domain(domain_name, name, data_type)?;
            let (target_name, indices) = domain_mapping(
                domain,
                point_face_sources,
                face_point_sources,
                edge_sources,
                corner_sources,
            );
            let values = entry
                .get("values")
                .and_then(Value::as_array)
                .ok_or_else(|| {
                    invalid(format!(
                        "Dual Mesh Blender attribute `{name}` values must be an array"
                    ))
                })?;
            let remapped = gather_array(
                values,
                indices,
                source_count(source, domain, source_corner_count),
                name,
                data_type,
            )?;
            let mut result = entry
                .as_object()
                .cloned()
                .ok_or_else(|| invalid("Dual Mesh Blender attribute must be an object"))?;
            result.insert(
                "domain".to_owned(),
                json!(target_domain_name(domain_name, target_name)),
            );
            result.insert("values".to_owned(), Value::Array(remapped));
            Ok(Some(Value::Object(result)))
        })
        .filter_map(|entry| match entry {
            Ok(Some(value)) => Some(Ok(value)),
            Ok(None) => None,
            Err(error) => Some(Err(error)),
        })
        .collect::<Result<Vec<_>>>()
        .map(Value::Array)
}

fn is_builtin_excluded(name: &str) -> bool {
    matches!(
        name,
        "position" | ".edge_verts" | ".corner_vert" | ".corner_edge" | "sharp_face"
    )
}

fn remap_edge_flags(value: &Value, source: &Mesh, edge_sources: &[Option<usize>]) -> Result<Value> {
    let flags = value
        .as_array()
        .filter(|flags| flags.len() == source.edges.len())
        .ok_or_else(|| invalid("Dual Mesh edge flags do not match the source edges"))?;
    edge_sources
        .iter()
        .map(|source_index| match source_index {
            Some(index) => flags
                .get(*index)
                .filter(|flag| flag.is_object())
                .cloned()
                .ok_or_else(|| invalid("Dual Mesh source edge flags are malformed")),
            None => Ok(json!({"sharp": false, "seam": false})),
        })
        .collect::<Result<Vec<_>>>()
        .map(Value::Array)
}

fn remap_polygon_smooth(value: &Value, source: &Mesh, output: &Mesh) -> Result<Value> {
    let smooth = value
        .as_array()
        .filter(|smooth| smooth.len() == source.faces.len())
        .ok_or_else(|| invalid("Dual Mesh polygon smooth flags do not match the source faces"))?;
    if smooth.iter().any(|flag| !flag.is_boolean()) {
        return Err(invalid("Dual Mesh source polygon smooth flag is malformed"));
    }
    Ok(Value::Array(
        std::iter::repeat_n(json!(false), output.faces.len()).collect(),
    ))
}
fn remap_shade_smooth(value: &Value, source: &Mesh, output: &Mesh) -> Result<Value> {
    let source_values = value
        .get("values")
        .and_then(Value::as_object)
        .ok_or_else(|| invalid("Dual Mesh shade-smooth values are malformed"))?;
    for face in &source.faces {
        if !source_values
            .get(&format!("f{}", face.id))
            .is_some_and(Value::is_boolean)
        {
            return Err(invalid(
                "Dual Mesh shade-smooth values do not match the source faces",
            ));
        }
    }
    let mut result = value
        .as_object()
        .cloned()
        .ok_or_else(|| invalid("Dual Mesh shade-smooth data must be an object"))?;
    let values = output
        .faces
        .iter()
        .map(|face| (format!("f{}", face.id), json!(false)))
        .collect();
    result.insert("domain".to_owned(), json!("face"));
    result.insert("values".to_owned(), Value::Object(values));
    Ok(Value::Object(result))
}

fn remap_origindex(value: &Value, source: &Mesh, output: &Mesh) -> Result<Value> {
    let indices = value
        .as_array()
        .filter(|indices| indices.len() == source.vertices.len())
        .ok_or_else(|| invalid("Dual Mesh origindex data does not match the source points"))?;
    if indices.iter().any(|index| {
        index
            .as_i64()
            .and_then(|index| i32::try_from(index).ok())
            .is_none()
    }) {
        return Err(invalid(
            "Dual Mesh origindex contains a value outside Blender's integer range",
        ));
    }
    Ok(Value::Array(
        std::iter::repeat_n(json!(ORIGINDEX_NONE), output.vertices.len()).collect(),
    ))
}
fn remap_vertex_groups(
    source: &Mesh,
    face_point_sources: &[usize],
    attributes: &mut Map<String, Value>,
) -> Result<()> {
    let Some(groups) = source.attributes.get("vertex_groups") else {
        return Ok(());
    };
    let groups = groups
        .as_object()
        .ok_or_else(|| invalid("Dual Mesh vertex groups must be an object"))?;
    for (name, weights) in groups {
        if name.is_empty() {
            return Err(invalid("Dual Mesh vertex group name is empty"));
        }
        let weights = weights
            .as_object()
            .ok_or_else(|| invalid(format!("Dual Mesh vertex group `{name}` is malformed")))?;
        let values = face_point_sources
            .iter()
            .map(|source_index| {
                let source_vertex = source
                    .vertices
                    .get(*source_index)
                    .ok_or_else(|| invalid("Dual Mesh vertex group source is out of range"))?;
                let value = match weights.get(&format!("v{}", source_vertex.id)) {
                    Some(value) => value.as_f64().ok_or_else(|| {
                        invalid(format!(
                            "Dual Mesh vertex group `{name}` has a nonnumeric weight"
                        ))
                    })?,
                    None => 0.0,
                };
                Ok(json!(value))
            })
            .collect::<Result<Vec<_>>>()?;
        let key = attribute_keys::storage_key(name).into_owned();
        if attributes.contains_key(&key) {
            return Err(PotError::with_details(
                ErrorCode::UnsupportedFeature,
                format!("Dual Mesh cannot represent colliding vertex group `{name}`"),
                json!({"attribute": name}),
            ));
        }
        attributes.insert(
            key,
            json!({"domain":"FACE", "data_type":"FLOAT", "data":values}),
        );
    }
    Ok(())
}

fn remap_colors(
    store: &ColorStore,
    output: &Mesh,
    point_face_sources: &[Option<usize>],
    face_point_sources: &[usize],
    corner_sources: &[usize],
) -> Result<ColorStore> {
    let mut result = ColorStore {
        layers: Vec::with_capacity(store.layers.len()),
        active: store.active.clone(),
        render: store.render.clone(),
    };
    for layer in &store.layers {
        let (domain, values) =
            match layer.domain {
                ColorDomain::Point => (
                    ColorDomain::Face,
                    face_point_sources
                        .iter()
                        .map(|&index| {
                            layer.values.get(index).copied().ok_or_else(|| {
                                invalid("Dual Mesh point color source is out of range")
                            })
                        })
                        .collect::<Result<Vec<_>>>()?,
                ),
                ColorDomain::Face => (
                    ColorDomain::Point,
                    point_face_sources
                        .iter()
                        .map(|index| {
                            index
                                .and_then(|index| layer.values.get(index))
                                .copied()
                                .unwrap_or([0.0; 4])
                        })
                        .collect(),
                ),
                ColorDomain::Corner => (
                    ColorDomain::Corner,
                    corner_sources
                        .iter()
                        .map(|&index| {
                            layer.values.get(index).copied().ok_or_else(|| {
                                invalid("Dual Mesh color corner source is out of range")
                            })
                        })
                        .collect::<Result<Vec<_>>>()?,
                ),
            };
        result.layers.push(ColorLayer {
            name: layer.name.clone(),
            domain,
            values,
        });
    }
    if result.layers.iter().any(|layer| match layer.domain {
        ColorDomain::Point => layer.values.len() != output.vertices.len(),
        ColorDomain::Face => layer.values.len() != output.faces.len(),
        ColorDomain::Corner => layer.values.len() != corner_sources.len(),
    }) {
        return Err(invalid(
            "Dual Mesh color transfer produced the wrong domain size",
        ));
    }
    Ok(result)
}

fn remap_uv_layers(
    source: &Mesh,
    output: &mut Mesh,
    corner_sources: &[usize],
    source_corners: &[(usize, usize)],
) -> Result<()> {
    let source_store = uv_layers::read(source);
    if source_store.layers.is_empty() {
        return Ok(());
    }
    let mut output_store = UvStore {
        layers: Vec::with_capacity(source_store.layers.len()),
        active: source_store.active,
        render: source_store.render,
    };
    for layer in source_store.layers {
        let mut faces = Vec::with_capacity(output.faces.len());
        let mut pinned = Vec::with_capacity(output.faces.len());
        let mut start = 0;
        for face in &output.faces {
            let count = face.vertices.len();
            let end = start + count;
            let mut uv = Vec::with_capacity(count);
            let mut pins = Vec::with_capacity(count);
            let has_pins = layer.pinned.iter().any(Option::is_some);
            for &source_corner_index in &corner_sources[start..end] {
                let (source_face, source_corner) = *source_corners
                    .get(source_corner_index)
                    .ok_or_else(|| invalid("Dual Mesh source UV corner is out of range"))?;
                uv.push(
                    layer
                        .faces
                        .get(source_face)
                        .and_then(Option::as_ref)
                        .and_then(|values| values.get(source_corner))
                        .copied()
                        .unwrap_or([0.0, 1.0]),
                );
                if has_pins {
                    pins.push(
                        layer
                            .pinned
                            .get(source_face)
                            .and_then(Option::as_ref)
                            .and_then(|values| values.get(source_corner))
                            .copied()
                            .unwrap_or(false),
                    );
                }
            }
            faces.push(Some(uv));
            pinned.push(has_pins.then_some(pins));
            start = end;
        }
        output_store.layers.push(UvLayer {
            name: layer.name,
            faces,
            pinned,
        });
    }
    uv_layers::write(output, &output_store);
    Ok(())
}

fn corner_count(mesh: &Mesh) -> Result<usize> {
    mesh.faces.iter().try_fold(0_usize, |count, face| {
        count.checked_add(face.vertices.len()).ok_or_else(|| {
            PotError::new(
                ErrorCode::LimitExceeded,
                "Dual Mesh corner count overflows usize",
            )
        })
    })
}

fn corner_locations(mesh: &Mesh) -> Result<Vec<(usize, usize)>> {
    let mut result = Vec::with_capacity(corner_count(mesh)?);
    for (face_index, face) in mesh.faces.iter().enumerate() {
        result.extend((0..face.vertices.len()).map(|corner| (face_index, corner)));
    }
    Ok(result)
}

fn invalid(message: impl Into<String>) -> PotError {
    PotError::new(ErrorCode::InternalError, message)
}

#[cfg(test)]
mod tests {
    use std::error::Error;

    use glam::DVec3;
    use serde_json::json;

    use crate::geom::{
        Mesh,
        color_layers::{self, ColorDomain, ColorLayer, ColorStore},
        set_origindex,
        uv_layers::{self, UvLayer, UvStore},
    };

    use super::transfer;

    fn triangle() -> std::result::Result<Mesh, Box<dyn Error>> {
        Ok(Mesh::from_positions_and_faces(
            vec![DVec3::ZERO, DVec3::X, DVec3::Y],
            vec![vec![0, 1, 2]],
        )?)
    }

    #[test]
    fn gathers_cross_domain_and_corner_stores() -> std::result::Result<(), Box<dyn Error>> {
        let mut source = triangle()?;
        source.attributes.insert(
            "weight".to_owned(),
            json!({"domain":"point","type":"float","values":{"v0":0.25,"v1":0.5,"v2":0.75}}),
        );
        source.attributes.insert(
            "material_index".to_owned(),
            json!({"domain":"face","type":"int","values":{"f0":7}}),
        );
        source.attributes.insert(
            "edge_data".to_owned(),
            json!({"domain":"edge","type":"int","values":{"e0":10,"e1":20,"e2":30}}),
        );
        source.attributes.insert(
            "corner_data".to_owned(),
            json!({"domain":"corner","type":"float","values":{"f0":[1.0,2.0,3.0]}}),
        );
        source.attributes.insert(
            "blender_attributes".to_owned(),
            json!([
                {"name":"face_value","domain":"FACE","data_type":"INT","values":[11]},
                {"name":"point_value","domain":"POINT","data_type":"FLOAT","values":[1.0,2.0,3.0]},
                {"name":"edge_value","domain":"EDGE","data_type":"INT","values":[4,5,6]},
                {"name":"corner_value","domain":"CORNER","data_type":"INT","values":[7,8,9]}
            ]),
        );
        source.attributes.insert(
            "blender_metadata".to_owned(),
            json!({"color_data_types":{"PointPaint":"BYTE_COLOR"}}),
        );
        let colors = ColorStore {
            layers: vec![
                ColorLayer {
                    name: "PointPaint".to_owned(),
                    domain: ColorDomain::Point,
                    values: vec![
                        [0.1, 0.2, 0.3, 1.0],
                        [0.4, 0.5, 0.6, 1.0],
                        [0.7, 0.8, 0.9, 1.0],
                    ],
                },
                ColorLayer {
                    name: "CornerPaint".to_owned(),
                    domain: ColorDomain::Corner,
                    values: vec![
                        [1.0, 0.0, 0.0, 1.0],
                        [0.0, 1.0, 0.0, 1.0],
                        [0.0, 0.0, 1.0, 1.0],
                    ],
                },
                ColorLayer {
                    name: "FacePaint".to_owned(),
                    domain: ColorDomain::Face,
                    values: vec![[0.3, 0.4, 0.5, 0.6]],
                },
            ],
            active: Some("PointPaint".to_owned()),
            render: Some("CornerPaint".to_owned()),
        };
        color_layers::write(&mut source, &colors);
        let uv = UvStore {
            layers: vec![UvLayer {
                name: "UVMap".to_owned(),
                faces: vec![Some(vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])],
                pinned: vec![Some(vec![true, false, true])],
            }],
            active: Some("UVMap".to_owned()),
            render: Some("UVMap".to_owned()),
        };
        uv_layers::write(&mut source, &uv);
        set_origindex(&mut source, &[2, 3, 4]);

        let mut output = triangle()?;
        transfer(
            &source,
            &mut output,
            &[Some(0), None, Some(0)],
            &[2],
            &[Some(2), Some(1), None],
            &[2, 1, 0],
        )?;

        assert_eq!(output.attributes["weight"]["domain"], "face");
        assert_eq!(output.attributes["weight"]["values"]["f0"], 0.75);
        assert_eq!(output.attributes["material_index"]["domain"], "point");
        assert_eq!(output.attributes["material_index"]["values"]["v1"], 0);
        assert_eq!(output.attributes["edge_data"]["values"]["e0"], 30);
        assert_eq!(output.attributes["edge_data"]["values"]["e2"], 0);
        assert_eq!(
            output.attributes["corner_data"]["values"]["f0"],
            json!([3.0, 2.0, 1.0])
        );
        assert_eq!(
            output.attributes["blender_attributes"][0]["domain"],
            "POINT"
        );
        assert_eq!(
            output.attributes["blender_attributes"][0]["values"],
            json!([11, 0, 11])
        );
        assert_eq!(output.attributes["blender_attributes"][1]["domain"], "FACE");
        assert_eq!(
            output.attributes["blender_attributes"][1]["values"],
            json!([3.0])
        );
        assert_eq!(
            output.attributes["blender_attributes"][2]["values"],
            json!([6, 5, 0])
        );
        assert_eq!(
            output.attributes["blender_attributes"][3]["values"],
            json!([9, 8, 7])
        );
        let colors = color_layers::read(&output);
        assert_eq!(colors.layers[0].domain, ColorDomain::Face);
        assert_eq!(colors.layers[0].values, [[0.7, 0.8, 0.9, 1.0]]);
        assert_eq!(
            colors.layers[1].values,
            [
                [0.0, 0.0, 1.0, 1.0],
                [0.0, 1.0, 0.0, 1.0],
                [1.0, 0.0, 0.0, 1.0]
            ]
        );
        assert_eq!(colors.layers[2].domain, ColorDomain::Point);
        assert_eq!(
            colors.layers[2].values,
            [[0.3, 0.4, 0.5, 0.6], [0.0; 4], [0.3, 0.4, 0.5, 0.6]]
        );
        assert_eq!(colors.active.as_deref(), Some("PointPaint"));
        assert_eq!(colors.render.as_deref(), Some("CornerPaint"));
        assert_eq!(
            output.attributes["blender_metadata"]["color_data_types"]["PointPaint"],
            "BYTE_COLOR"
        );
        assert_eq!(
            uv_layers::read(&output).layers[0].faces[0],
            Some(vec![[0.0, 1.0], [1.0, 0.0], [0.0, 0.0]])
        );
        assert_eq!(
            uv_layers::read(&output).layers[0].pinned[0],
            Some(vec![true, false, true])
        );
        color_layers::validate(&output)?;
        uv_layers::validate(&output)?;
        assert_eq!(
            output.attributes["user:__potter_origindex"],
            json!([-1, -1, -1])
        );
        Ok(())
    }
}
