use std::collections::{BTreeMap, BTreeSet};

use glam::{DVec2, DVec3};
use serde_json::{Map, Value, json};

use crate::{
    error::{ErrorCode, PotError, Result},
    geom::{
        Mesh,
        attribute_keys::{storage_key, user_name},
        color_layers::{self, ColorDomain, ColorLayer},
        sculpt as sculpt_geom, uv_layers,
    },
    image::{ImageData, ImageInterpolation, encode_pixels, load_image_data_with_staged},
    model::{DataBlock, Id, ImageSource, ImageTile},
};

use super::{ChangeKind, Engine, check_fields};

pub(super) fn apply(
    engine: &mut Engine<'_>,
    name: &str,
    operation: &Map<String, Value>,
) -> Result<bool> {
    match name {
        "paint.vertex" => vertex(engine, operation),
        "paint.weight" => weight(engine, operation),
        "paint.texture" => texture(engine, operation),
        _ => Err(engine.error(
            ErrorCode::InvalidOperation,
            format!("unsupported paint operation `{name}`"),
            "/op",
        )),
    }
}

fn vertex(engine: &mut Engine<'_>, operation: &Map<String, Value>) -> Result<bool> {
    check_fields(
        engine,
        operation,
        &[
            "op", "target", "scope", "samples", "falloff", "symmetry", "color", "blend",
        ],
        &["target", "samples", "color"],
    )?;
    let color = color_value(engine, operation.get("color"), "/color")?;
    let blend = match operation.get("blend") {
        None => "mix",
        Some(Value::String(blend)) => blend.as_str(),
        Some(_) => {
            return Err(engine.error(
                ErrorCode::InvalidOperation,
                "paint.vertex blend must be a string",
                "/blend",
            ));
        }
    };
    if !["mix", "add", "multiply"].contains(&blend) {
        return Err(engine.error(
            ErrorCode::InvalidOperation,
            "paint.vertex blend must be mix, add, or multiply",
            "/blend",
        ));
    }
    let stroke = parse_paint_stroke(engine, operation)?;
    let targets = super::sculpt::prepare_mesh_targets(engine, operation)?;
    let mut changed = false;
    for (node_id, data_id) in targets {
        let world = super::sculpt::node_world_matrix(engine.doc, &node_id)?;
        let before = engine
            .doc
            .data_blocks
            .get(&data_id)
            .and_then(|block| block.mesh.as_ref())
            .cloned()
            .ok_or_else(|| {
                engine.error(
                    ErrorCode::InvalidOperation,
                    "target data has no mesh",
                    "/target",
                )
            })?;
        let mut world_mesh = before;
        for vertex in &mut world_mesh.vertices {
            vertex.co = world.transform_point3(vertex.co);
        }
        let influences = sculpt_geom::influences(&world_mesh, &stroke)?
            .into_iter()
            .collect::<std::collections::BTreeMap<_, _>>();
        let block = engine.doc.data_blocks.get_mut(&data_id).ok_or_else(|| {
            PotError::new(ErrorCode::TargetNotFound, "mesh data block disappeared")
        })?;
        let mesh = block
            .mesh
            .as_mut()
            .ok_or_else(|| PotError::new(ErrorCode::InvalidOperation, "target data has no mesh"))?;
        if paint_vertex_colors(mesh, &influences, color, blend) {
            engine.mark("data_blocks", &data_id, ChangeKind::Updated);
            changed = true;
        }
    }
    Ok(changed)
}

fn weight(engine: &mut Engine<'_>, operation: &Map<String, Value>) -> Result<bool> {
    check_fields(
        engine,
        operation,
        &[
            "op",
            "target",
            "scope",
            "samples",
            "falloff",
            "symmetry",
            "group",
            "weight",
            "normalize",
            "lock",
        ],
        &["target", "samples", "group", "weight"],
    )?;
    let group = super::read_id(engine, operation, "group")?;
    let weight_value = operation
        .get("weight")
        .and_then(Value::as_f64)
        .filter(|weight| weight.is_finite() && (0.0..=1.0).contains(weight))
        .ok_or_else(|| {
            engine.error(
                ErrorCode::InvalidOperation,
                "weight must be between zero and one",
                "/weight",
            )
        })?;
    let normalize = operation
        .get("normalize")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let locked = operation
        .get("lock")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    if operation
        .get("normalize")
        .is_some_and(|value| !value.is_boolean())
        || operation
            .get("lock")
            .is_some_and(|value| !value.is_boolean())
    {
        return Err(engine.error(
            ErrorCode::InvalidOperation,
            "normalize and lock must be booleans",
            "/normalize",
        ));
    }
    let stroke = parse_paint_stroke(engine, operation)?;
    let targets = super::sculpt::prepare_mesh_targets(engine, operation)?;
    let mut changed = false;
    for (node_id, data_id) in targets {
        if locked {
            continue;
        }
        let world = super::sculpt::node_world_matrix(engine.doc, &node_id)?;
        let before = engine
            .doc
            .data_blocks
            .get(&data_id)
            .and_then(|block| block.mesh.as_ref())
            .cloned()
            .ok_or_else(|| {
                engine.error(
                    ErrorCode::InvalidOperation,
                    "target data has no mesh",
                    "/target",
                )
            })?;
        let mut world_mesh = before;
        for vertex in &mut world_mesh.vertices {
            vertex.co = world.transform_point3(vertex.co);
        }
        let influences = sculpt_geom::influences(&world_mesh, &stroke)?
            .into_iter()
            .collect::<std::collections::BTreeMap<_, _>>();
        let affected = world_mesh
            .vertices
            .iter()
            .filter_map(|vertex| {
                let amount = influences
                    .get(&vertex.id)
                    .copied()
                    .unwrap_or(0.0)
                    .clamp(0.0, 1.0);
                (amount > 0.0).then_some((vertex.id, amount))
            })
            .collect::<Vec<_>>();
        if affected.is_empty() {
            continue;
        }
        let block = engine.doc.data_blocks.get_mut(&data_id).ok_or_else(|| {
            PotError::new(ErrorCode::TargetNotFound, "mesh data block disappeared")
        })?;
        let data_changed = if block
            .vertex_groups
            .iter()
            .any(|vertex_group| vertex_group.id == group)
        {
            paint_group_weights(block, &group, &affected, weight_value, normalize)
        } else {
            let mesh = block.mesh.as_mut().ok_or_else(|| {
                PotError::new(ErrorCode::InvalidOperation, "target data has no mesh")
            })?;
            paint_weight_attribute(
                mesh,
                &format!("weight:{group}"),
                &affected,
                weight_value,
                normalize,
            )?
        };
        if data_changed {
            engine.mark("data_blocks", &data_id, ChangeKind::Updated);
            changed = true;
        }
    }
    Ok(changed)
}

fn parse_paint_stroke(
    engine: &Engine<'_>,
    operation: &Map<String, Value>,
) -> Result<sculpt_geom::Stroke> {
    let mut fields = operation.clone();
    fields.insert("brush".to_owned(), Value::String("draw".to_owned()));
    super::sculpt::parse_stroke(engine, &fields)
}

fn color_value(engine: &Engine<'_>, value: Option<&Value>, pointer: &str) -> Result<[f64; 4]> {
    let values = value
        .and_then(Value::as_array)
        .filter(|values| values.len() == 4)
        .ok_or_else(|| {
            engine.error(
                ErrorCode::InvalidOperation,
                "color must be four linear RGBA components",
                pointer,
            )
        })?;
    let [r, g, b, a] = [0, 1, 2, 3].map(|index| {
        values[index]
            .as_f64()
            .filter(|component| component.is_finite() && (0.0..=1.0).contains(component))
    });
    match [r, g, b, a] {
        [Some(r), Some(g), Some(b), Some(a)] => Ok([r, g, b, a]),
        _ => Err(engine.error(
            ErrorCode::InvalidOperation,
            "color components must be between zero and one",
            pointer,
        )),
    }
}

/// The color layer that vertex painting writes. Blender names its default paint layer the same way.
const PAINT_LAYER: &str = "color";

/// Blender's default color: opaque black. Elements a new layer has not painted take it.
const DEFAULT_COLOR: [f64; 4] = [0.0, 0.0, 0.0, 1.0];

/// Paints `color` onto the `color` layer of `mesh`, creating a corner layer when there is none. Each
/// vertex's influence blends into its values: a corner layer takes it at every face corner of that
/// vertex, and a point layer takes it at the vertex. The store is written only when a value changed.
/// Returns whether any value changed.
fn paint_vertex_colors(
    mesh: &mut Mesh,
    influences: &BTreeMap<u32, f64>,
    color: [f64; 4],
    blend: &str,
) -> bool {
    let mut store = color_layers::read(mesh);
    let index = store.index_of(PAINT_LAYER).unwrap_or_else(|| {
        let corners = mesh.faces.iter().map(|face| face.vertices.len()).sum();
        store.layers.push(ColorLayer {
            name: PAINT_LAYER.to_owned(),
            domain: ColorDomain::Corner,
            values: vec![DEFAULT_COLOR; corners],
        });
        store.layers.len() - 1
    });
    let layer = &mut store.layers[index];
    let mut changed = false;
    match layer.domain {
        ColorDomain::Point => {
            for (vertex, value) in mesh.vertices.iter().zip(layer.values.iter_mut()) {
                let amount = influence(influences, vertex.id);
                if amount > 0.0 {
                    changed |= paint_value(value, color, amount, blend);
                }
            }
        }
        ColorDomain::Corner => {
            let corner_vertices = mesh.faces.iter().flat_map(|face| face.vertices.iter());
            for (vertex_id, value) in corner_vertices.zip(layer.values.iter_mut()) {
                let amount = influence(influences, *vertex_id);
                if amount > 0.0 {
                    changed |= paint_value(value, color, amount, blend);
                }
            }
        }
    }
    if changed {
        color_layers::write(mesh, &store);
    }
    changed
}

/// The clamped paint influence of vertex `id`.
fn influence(influences: &BTreeMap<u32, f64>, id: u32) -> f64 {
    influences.get(&id).copied().unwrap_or(0.0).clamp(0.0, 1.0)
}

/// Blends `color` into `current` by `amount`. Returns whether a channel changed.
fn paint_value(current: &mut [f64; 4], color: [f64; 4], amount: f64, blend: &str) -> bool {
    let painted = blend_color(*current, color, amount, blend);
    let changed = current
        .iter()
        .zip(painted)
        .any(|(before, after)| (*before - after).abs() > f64::EPSILON);
    if changed {
        *current = painted;
    }
    changed
}

fn blend_color(current: [f64; 4], color: [f64; 4], amount: f64, mode: &str) -> [f64; 4] {
    std::array::from_fn(|index| {
        let result = match mode {
            "mix" => current[index] + (color[index] - current[index]) * amount,
            "add" => current[index] + color[index] * amount,
            "multiply" => current[index] * (1.0 + (color[index] - 1.0) * amount),
            _ => current[index],
        };
        result.clamp(0.0, 1.0)
    })
}

fn attribute_values<'a>(mesh: &'a mut Mesh, name: &str, domain: &str) -> Result<&'a mut Value> {
    let storage = storage_key(name);
    super::sculpt::attribute_values(mesh, &storage, domain)
}
fn paint_weight_attribute(
    mesh: &mut Mesh,
    name: &str,
    affected: &[(u32, f64)],
    weight: f64,
    normalize: bool,
) -> Result<bool> {
    let mut changed = {
        let values = attribute_values(mesh, name, "vertices")?;
        let values = values.as_object_mut().ok_or_else(|| {
            PotError::invalid_argument("weight attribute values must be an object")
        })?;
        let mut changed = false;
        for (vertex_id, amount) in affected {
            let key = format!("v{vertex_id}");
            let current = values.get(&key).and_then(Value::as_f64).unwrap_or(0.0);
            let next = current + (weight - current) * amount;
            if (current - next).abs() > f64::EPSILON {
                values.insert(key, json!(next));
                changed = true;
            }
        }
        changed
    };
    if normalize {
        let touched = affected
            .iter()
            .map(|(vertex_id, _)| *vertex_id)
            .collect::<Vec<_>>();
        changed |= normalize_vertex_weights(mesh, &touched);
    }
    Ok(changed)
}

fn paint_group_weights(
    block: &mut DataBlock,
    group: &Id,
    affected: &[(u32, f64)],
    weight: f64,
    normalize: bool,
) -> bool {
    let mut changed = false;
    for (vertex_id, amount) in affected {
        let values = block.vertex_weights.entry(*vertex_id).or_default();
        let current = values.get(group).copied().unwrap_or(0.0);
        let next = current + (weight - current) * amount;
        if (current - next).abs() > f64::EPSILON {
            values.insert(group.clone(), next);
            changed = true;
        }
    }
    if normalize {
        for (vertex_id, _) in affected {
            let Some(values) = block.vertex_weights.get_mut(vertex_id) else {
                continue;
            };
            let total = values.values().sum::<f64>();
            if total <= f64::EPSILON {
                continue;
            }
            for value in values.values_mut() {
                let normalized = *value / total;
                if (*value - normalized).abs() > f64::EPSILON {
                    *value = normalized;
                    changed = true;
                }
            }
        }
    }
    changed
}

fn normalize_vertex_weights(mesh: &mut Mesh, touched: &[u32]) -> bool {
    let names = mesh
        .attributes
        .keys()
        .filter_map(|key| {
            let name = user_name(key)?;
            name.starts_with("weight:").then(|| key.clone())
        })
        .collect::<Vec<_>>();
    let mut changed = false;
    for vertex_id in touched {
        let key = format!("v{vertex_id}");
        let total = names
            .iter()
            .filter_map(|name| {
                mesh.attributes
                    .get(name)?
                    .get("values")?
                    .get(&key)?
                    .as_f64()
            })
            .sum::<f64>();
        if total <= f64::EPSILON {
            continue;
        }
        for name in &names {
            if let Some(value) = mesh
                .attributes
                .get_mut(name)
                .and_then(|attribute| attribute.get_mut("values"))
                .and_then(Value::as_object_mut)
                .and_then(|values| values.get_mut(&key))
                && let Some(weight) = value.as_f64()
            {
                let normalized = weight / total;
                if (weight - normalized).abs() > f64::EPSILON {
                    *value = json!(normalized);
                    changed = true;
                }
            }
        }
    }
    changed
}
fn texture(engine: &mut Engine<'_>, operation: &Map<String, Value>) -> Result<bool> {
    check_fields(
        engine,
        operation,
        &[
            "op", "target", "scope", "image", "uv_layer", "samples", "falloff", "symmetry",
            "color", "blend",
        ],
        &["target", "image", "samples", "color"],
    )?;
    let image_id = super::read_id(engine, operation, "image")?;
    let image = engine.doc.images.get(&image_id).cloned().ok_or_else(|| {
        engine.error(
            ErrorCode::TargetNotFound,
            format!("image `{image_id}` was not found"),
            "/image",
        )
    })?;
    let root = engine.asset_root.ok_or_else(|| {
        engine.error(
            ErrorCode::InvalidOperation,
            "texture painting requires a project asset context",
            "/op",
        )
    })?;
    let mut image_data = load_image_data_with_staged(
        &image,
        root,
        ImageInterpolation::Closest,
        &engine.pending_assets,
    )?;
    let color = color_value(engine, operation.get("color"), "/color")?;
    let blend = match operation.get("blend") {
        None => "mix",
        Some(Value::String(blend)) => blend.as_str(),
        Some(_) => {
            return Err(engine.error(
                ErrorCode::InvalidOperation,
                "paint.texture blend must be a string",
                "/blend",
            ));
        }
    };
    if !["mix", "add", "multiply"].contains(&blend) {
        return Err(engine.error(
            ErrorCode::InvalidOperation,
            "paint.texture blend must be mix, add, or multiply",
            "/blend",
        ));
    }
    let uv_layer = match operation.get("uv_layer") {
        None => None,
        Some(Value::String(name)) if name.is_empty() => {
            return Err(engine.error(
                ErrorCode::InvalidOperation,
                "uv_layer must not be empty; omit it to use the active layer",
                "/uv_layer",
            ));
        }
        Some(Value::String(name)) => Some(name.as_str()),
        Some(_) => {
            return Err(engine.error(
                ErrorCode::InvalidOperation,
                "uv_layer must be a string",
                "/uv_layer",
            ));
        }
    };
    let stroke = parse_paint_stroke(engine, operation)?;
    let symmetry_mask = stroke_symmetry_mask(&stroke);
    let targets = super::sculpt::prepare_mesh_targets(engine, operation)?;
    let mut influences = BTreeMap::<(u32, u32, u32), f64>::new();
    for (node_id, data_id) in targets {
        let world = super::sculpt::node_world_matrix(engine.doc, &node_id)?;
        let mesh = engine
            .doc
            .data_blocks
            .get(&data_id)
            .and_then(|block| block.mesh.as_ref())
            .cloned()
            .ok_or_else(|| {
                engine.error(
                    ErrorCode::InvalidOperation,
                    "texture paint target has no mesh",
                    "/target",
                )
            })?;
        let uv_faces = read_corner_uvs(engine, &mesh, uv_layer)?;
        collect_texture_influences(
            &mesh,
            world,
            &uv_faces,
            &image_data,
            &stroke,
            symmetry_mask,
            &mut influences,
        )?;
    }
    if influences.is_empty() {
        return Ok(false);
    }
    let mut dirty_tiles = BTreeSet::new();
    for ((tile_number, y, x), amount) in influences {
        let (width, pixels) = if tile_number == 1001 {
            (image_data.width, &mut image_data.pixels)
        } else if let Some(tile) = image_data.tiles.get_mut(&tile_number) {
            (tile.width, &mut tile.pixels)
        } else {
            continue;
        };
        let row = usize::try_from(y).map_err(|_| {
            PotError::new(
                ErrorCode::LimitExceeded,
                "paint texel index exceeds platform limits",
            )
        })?;
        let column = usize::try_from(x).map_err(|_| {
            PotError::new(
                ErrorCode::LimitExceeded,
                "paint texel index exceeds platform limits",
            )
        })?;
        let stride = usize::try_from(width).map_err(|_| {
            PotError::new(
                ErrorCode::LimitExceeded,
                "image width exceeds platform limits",
            )
        })?;
        let index = row
            .checked_mul(stride)
            .and_then(|offset| offset.checked_add(column))
            .ok_or_else(|| {
                PotError::new(
                    ErrorCode::LimitExceeded,
                    "paint texel offset exceeds platform limits",
                )
            })?;
        let Some(current) = pixels.get_mut(index) else {
            return Err(PotError::new(
                ErrorCode::SceneInvalid,
                "paint texel lies outside its image tile",
            ));
        };
        let painted = blend_color(*current, color, amount, blend);
        if !crate::float::equal_f64_array(current, &painted) {
            *current = painted;
            dirty_tiles.insert(tile_number);
        }
    }
    if dirty_tiles.is_empty() {
        return Ok(false);
    }
    let mut updated = image;
    for tile_number in dirty_tiles {
        if tile_number == 1001 {
            let bytes = encode_pixels(image_data.width, image_data.height, &image_data.pixels)?;
            updated.blob = Some(stage_image_blob(engine, bytes));
        } else {
            let tile = image_data.tiles.get(&tile_number).ok_or_else(|| {
                PotError::new(ErrorCode::InternalError, "painted UDIM tile disappeared")
            })?;
            let bytes = encode_pixels(tile.width, tile.height, &tile.pixels)?;
            let blob = stage_image_blob(engine, bytes);
            if let Some(metadata) = updated
                .tiles
                .iter_mut()
                .find(|entry| entry.number == tile_number)
            {
                metadata.blob = blob;
            } else {
                updated.tiles.push(ImageTile {
                    number: tile_number,
                    width: tile.width,
                    height: tile.height,
                    blob,
                });
                updated.tiles.sort_by_key(|entry| entry.number);
            }
        }
    }
    if updated.blob.is_none() {
        let bytes = encode_pixels(image_data.width, image_data.height, &image_data.pixels)?;
        updated.blob = Some(stage_image_blob(engine, bytes));
    }
    updated.source = ImageSource::Packed;
    updated.source_path = None;
    updated.source_hash = None;
    engine.doc.images.insert(image_id.clone(), updated);
    engine.mark("images", &image_id, ChangeKind::Updated);
    Ok(true)
}

/// Corner UVs of the layer `requested` names, or of the active layer when `requested` is `None`.
/// Keyed by face ID; faces without an entry are absent. A named layer that does not exist is an
/// error: unlike the render path, painting never falls back to another layer.
fn read_corner_uvs(
    engine: &Engine<'_>,
    mesh: &Mesh,
    requested: Option<&str>,
) -> Result<BTreeMap<u32, Vec<DVec2>>> {
    let store = uv_layers::read(mesh);
    let index = match requested {
        None => store
            .active
            .as_deref()
            .and_then(|active| store.index_of(active)),
        Some(name) => store.index_of(name),
    };
    let layer = index
        .and_then(|index| store.layers.get(index))
        .ok_or_else(|| super::uv::layer_missing(requested))?;
    check_unique_face_rows(engine, mesh, &layer.name)?;
    let mut faces = BTreeMap::new();
    for (face, values) in mesh.faces.iter().zip(&layer.faces) {
        let Some(values) = values else {
            continue;
        };
        let mut corners = Vec::with_capacity(values.len());
        for uv in values {
            if !uv[0].is_finite() || !uv[1].is_finite() {
                return Err(engine.error(
                    ErrorCode::SceneInvalid,
                    "UV coordinate must be finite",
                    "/uv_layer",
                ));
            }
            corners.push(DVec2::new(uv[0], uv[1]));
        }
        faces.insert(face.id, corners);
    }
    Ok(faces)
}

/// Rejects a stored layer whose rows repeat a face ID. `uv_layers::read` keeps one row per face,
/// so a repeated row would otherwise be dropped without a trace.
fn check_unique_face_rows(engine: &Engine<'_>, mesh: &Mesh, layer_name: &str) -> Result<()> {
    let rows = mesh
        .attributes
        .get(uv_layers::ATTRIBUTE)
        .and_then(|store| store.get("layers"))
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .find(|layer| layer.get("name").and_then(Value::as_str) == Some(layer_name))
        .and_then(|layer| layer.get("faces"))
        .and_then(Value::as_array);
    let mut seen = BTreeSet::new();
    for entry in rows.into_iter().flatten() {
        let face_id = entry.get("face_id").and_then(Value::as_u64);
        if !face_id.is_some_and(|id| seen.insert(id)) {
            return Err(engine.error(
                ErrorCode::SceneInvalid,
                "UV map repeats a face ID",
                "/uv_layer",
            ));
        }
    }
    Ok(())
}

fn collect_texture_influences(
    mesh: &Mesh,
    world: glam::DMat4,
    uv_faces: &BTreeMap<u32, Vec<DVec2>>,
    image: &ImageData,
    stroke: &sculpt_geom::Stroke,
    symmetry_mask: u8,
    influences: &mut BTreeMap<(u32, u32, u32), f64>,
) -> Result<()> {
    let vertices = mesh
        .vertices
        .iter()
        .map(|vertex| (vertex.id, world.transform_point3(vertex.co)))
        .collect::<BTreeMap<_, _>>();
    for face in &mesh.faces {
        let Some(uvs) = uv_faces.get(&face.id) else {
            continue;
        };
        if uvs.len() != face.vertices.len() {
            return Err(PotError::new(
                ErrorCode::SceneInvalid,
                "face corner UV count does not match its vertex count",
            ));
        }
        if face.vertices.len() < 3 {
            continue;
        }
        let first_position = vertices.get(&face.vertices[0]).copied().ok_or_else(|| {
            PotError::new(
                ErrorCode::SceneInvalid,
                "UV face references a missing vertex",
            )
        })?;
        for index in 1..face.vertices.len() - 1 {
            let second_position =
                vertices
                    .get(&face.vertices[index])
                    .copied()
                    .ok_or_else(|| {
                        PotError::new(
                            ErrorCode::SceneInvalid,
                            "UV face references a missing vertex",
                        )
                    })?;
            let third_position = vertices
                .get(&face.vertices[index + 1])
                .copied()
                .ok_or_else(|| {
                    PotError::new(
                        ErrorCode::SceneInvalid,
                        "UV face references a missing vertex",
                    )
                })?;
            rasterize_uv_triangle(
                [uvs[0], uvs[index], uvs[index + 1]],
                [first_position, second_position, third_position],
                image,
                stroke,
                symmetry_mask,
                influences,
            )?;
        }
    }
    Ok(())
}

fn rasterize_uv_triangle(
    uv: [DVec2; 3],
    world: [DVec3; 3],
    image: &ImageData,
    stroke: &sculpt_geom::Stroke,
    symmetry_mask: u8,
    influences: &mut BTreeMap<(u32, u32, u32), f64>,
) -> Result<()> {
    let determinant = cross2(uv[1] - uv[0], uv[2] - uv[0]);
    if determinant.abs() <= f64::EPSILON {
        return Ok(());
    }
    let min_u = uv.iter().map(|point| point.x).fold(f64::INFINITY, f64::min);
    let max_u = uv
        .iter()
        .map(|point| point.x)
        .fold(f64::NEG_INFINITY, f64::max);
    let min_v = uv.iter().map(|point| point.y).fold(f64::INFINITY, f64::min);
    let max_v = uv
        .iter()
        .map(|point| point.y)
        .fold(f64::NEG_INFINITY, f64::max);
    let start_u = floor_to_i64(min_u).ok_or_else(|| {
        PotError::new(
            ErrorCode::InvalidOperation,
            "UV coordinate exceeds UDIM range",
        )
    })?;
    let start_v = floor_to_i64(min_v).ok_or_else(|| {
        PotError::new(
            ErrorCode::InvalidOperation,
            "UV coordinate exceeds UDIM range",
        )
    })?;
    let end_u = tile_end(max_u).ok_or_else(|| {
        PotError::new(
            ErrorCode::InvalidOperation,
            "UV coordinate exceeds UDIM range",
        )
    })?;
    let end_v = tile_end(max_v).ok_or_else(|| {
        PotError::new(
            ErrorCode::InvalidOperation,
            "UV coordinate exceeds UDIM range",
        )
    })?;
    if end_u < start_u || end_v < start_v {
        return Ok(());
    }
    if end_u.saturating_sub(start_u) > 128 || end_v.saturating_sub(start_v) > 128 {
        return Err(PotError::new(
            ErrorCode::LimitExceeded,
            "UV face spans too many UDIM tiles",
        ));
    }
    for tile_v in start_v..=end_v {
        for tile_u in start_u..=end_u {
            let Some(tile_number) = udim_number(tile_u, tile_v) else {
                continue;
            };
            let (width, height) = if tile_number == 1001 {
                (image.width, image.height)
            } else if let Some(tile) = image.tiles.get(&tile_number) {
                (tile.width, tile.height)
            } else {
                continue;
            };
            rasterize_triangle_tile(
                uv,
                world,
                [min_u, max_u, min_v, max_v],
                tile_u,
                tile_v,
                tile_number,
                width,
                height,
                stroke,
                symmetry_mask,
                influences,
            )?;
        }
    }
    Ok(())
}

#[expect(
    clippy::too_many_arguments,
    reason = "rasterization inputs are fixed triangle and image context data"
)]
fn rasterize_triangle_tile(
    uv: [DVec2; 3],
    world: [DVec3; 3],
    bounds: [f64; 4],
    tile_u: i64,
    tile_v: i64,
    tile_number: u32,
    width: u32,
    height: u32,
    stroke: &sculpt_geom::Stroke,
    symmetry_mask: u8,
    influences: &mut BTreeMap<(u32, u32, u32), f64>,
) -> Result<()> {
    let tile_min_u = f64::from(i32::try_from(tile_u).map_err(|_| {
        PotError::new(
            ErrorCode::LimitExceeded,
            "UDIM U coordinate exceeds supported range",
        )
    })?);
    let tile_min_v = f64::from(i32::try_from(tile_v).map_err(|_| {
        PotError::new(
            ErrorCode::LimitExceeded,
            "UDIM V coordinate exceeds supported range",
        )
    })?);
    let clipped_min_u = bounds[0].max(tile_min_u);
    let clipped_max_u = bounds[1].min(tile_min_u + 1.0);
    let clipped_min_v = bounds[2].max(tile_min_v);
    let clipped_max_v = bounds[3].min(tile_min_v + 1.0);
    if clipped_min_u >= clipped_max_u || clipped_min_v >= clipped_max_v {
        return Ok(());
    }
    let (x_start, x_end, y_start, y_end) = raster_bounds(
        clipped_min_u - tile_min_u,
        clipped_max_u - tile_min_u,
        clipped_min_v - tile_min_v,
        clipped_max_v - tile_min_v,
        width,
        height,
    )?;
    for y in y_start..=y_end {
        for x in x_start..=x_end {
            let point = DVec2::new(
                tile_min_u + (f64::from(x) + 0.5) / f64::from(width),
                tile_min_v + (f64::from(y) + 0.5) / f64::from(height),
            );
            let Some(weights) = barycentric(point, uv) else {
                continue;
            };
            if weights.iter().any(|weight| *weight < -1.0e-12) {
                continue;
            }
            let position = world[0] * weights[0] + world[1] * weights[1] + world[2] * weights[2];
            let amount = stroke_influence(position, stroke, symmetry_mask);
            if amount > 0.0 {
                influences
                    .entry((tile_number, y, x))
                    .and_modify(|existing| *existing = existing.max(amount))
                    .or_insert(amount);
            }
        }
    }
    Ok(())
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "raster UV bounds are clamped to finite image dimensions before conversion"
)]
fn raster_bounds(
    min_u: f64,
    max_u: f64,
    min_v: f64,
    max_v: f64,
    width: u32,
    height: u32,
) -> Result<(u32, u32, u32, u32)> {
    let max_x = f64::from(width - 1);
    let max_y = f64::from(height - 1);
    let x_start = (min_u * f64::from(width)).floor().clamp(0.0, max_x) as u32;
    let x_end = ((max_u * f64::from(width)).ceil() - 1.0).clamp(0.0, max_x) as u32;
    let y_start = (min_v * f64::from(height)).floor().clamp(0.0, max_y) as u32;
    let y_end = ((max_v * f64::from(height)).ceil() - 1.0).clamp(0.0, max_y) as u32;
    if x_start > x_end || y_start > y_end {
        return Err(PotError::new(
            ErrorCode::InvalidOperation,
            "UV raster bounds are empty",
        ));
    }
    Ok((x_start, x_end, y_start, y_end))
}

fn barycentric(point: DVec2, triangle: [DVec2; 3]) -> Option<[f64; 3]> {
    let first = triangle[1] - triangle[0];
    let second = triangle[2] - triangle[0];
    let relative = point - triangle[0];
    let determinant = cross2(first, second);
    (determinant.abs() > f64::EPSILON).then(|| {
        let weight_b = cross2(relative, second) / determinant;
        let weight_c = cross2(first, relative) / determinant;
        [1.0 - weight_b - weight_c, weight_b, weight_c]
    })
}

fn cross2(left: DVec2, right: DVec2) -> f64 {
    left.x * right.y - left.y * right.x
}

fn stroke_symmetry_mask(stroke: &sculpt_geom::Stroke) -> u8 {
    stroke.symmetry.iter().fold(0, |mask, axis| {
        mask | match axis.as_str() {
            "x" => 1,
            "y" => 2,
            "z" => 4,
            _ => 0,
        }
    })
}

fn stroke_influence(point: DVec3, stroke: &sculpt_geom::Stroke, symmetry_mask: u8) -> f64 {
    let variant_count = match symmetry_mask.count_ones() {
        0 => 1,
        1 => 2,
        2 => 4,
        _ => 8,
    };
    let mut maximum = 0.0_f64;
    for variant in 0..variant_count {
        if stroke.samples.len() == 1 {
            if let Some(sample) = stroke.samples.first() {
                let center = reflected(sample.position, symmetry_mask, variant);
                maximum = maximum.max(sample_influence(
                    point,
                    center,
                    sample.radius,
                    sample.pressure,
                    sample.strength,
                    stroke.falloff,
                ));
            }
        } else {
            for pair in stroke.samples.windows(2) {
                let first = &pair[0];
                let second = &pair[1];
                let from = reflected(first.position, symmetry_mask, variant);
                let to = reflected(second.position, symmetry_mask, variant);
                let segment = to - from;
                let length_squared = segment.length_squared();
                let t = if length_squared > 0.0 {
                    ((point - from).dot(segment) / length_squared).clamp(0.0, 1.0)
                } else {
                    0.0
                };
                maximum = maximum.max(sample_influence(
                    point,
                    from.lerp(to, t),
                    first.radius + (second.radius - first.radius) * t,
                    first.pressure + (second.pressure - first.pressure) * t,
                    first.strength + (second.strength - first.strength) * t,
                    stroke.falloff,
                ));
            }
        }
    }
    maximum
}

fn reflected(position: DVec3, symmetry_mask: u8, variant: usize) -> DVec3 {
    let mut result = position;
    let mut variant_bit = 0;
    for component in 0..3 {
        if symmetry_mask & (1_u8 << component) != 0 {
            if variant & (1_usize << variant_bit) != 0 {
                result[component] = -result[component];
            }
            variant_bit += 1;
        }
    }
    result
}

fn sample_influence(
    point: DVec3,
    center: DVec3,
    radius: f64,
    pressure: f64,
    strength: f64,
    falloff: sculpt_geom::Falloff,
) -> f64 {
    if radius <= 0.0 {
        return 0.0;
    }
    let normalized = point.distance(center) / radius;
    if normalized > 1.0 {
        0.0
    } else {
        (falloff.weight(normalized) * pressure * strength).clamp(0.0, 1.0)
    }
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "value is range-checked to fit i64 before conversion"
)]
fn floor_to_i64(value: f64) -> Option<i64> {
    const I64_MIN_AS_F64: f64 = -9_223_372_036_854_775_808.0;
    const I64_MAX_EXCLUSIVE_AS_F64: f64 = 9_223_372_036_854_775_808.0;
    let floor = value.floor();
    (value.is_finite() && (I64_MIN_AS_F64..I64_MAX_EXCLUSIVE_AS_F64).contains(&floor))
        .then_some(floor as i64)
}

fn tile_end(value: f64) -> Option<i64> {
    let floor = floor_to_i64(value)?;
    if crate::float::equal_f64(value, value.floor()) {
        floor.checked_sub(1)
    } else {
        Some(floor)
    }
}

fn udim_number(tile_u: i64, tile_v: i64) -> Option<u32> {
    let offset = tile_v.checked_mul(10)?.checked_add(tile_u)?;
    u32::try_from(i64::from(1001_u32).checked_add(offset)?).ok()
}

fn stage_image_blob(engine: &mut Engine<'_>, bytes: Vec<u8>) -> String {
    let digest = crate::hash::sha256(&bytes);
    engine.pending_assets.insert(digest.clone(), bytes);
    digest
}

#[cfg(test)]
mod tests {
    #![expect(clippy::unwrap_used, reason = "paint fixtures")]

    use std::collections::BTreeMap;

    use crate::geom::{
        BoxParams, Mesh,
        color_layers::{self, ColorDomain, ColorLayer},
    };

    use super::{DEFAULT_COLOR, paint_vertex_colors};

    fn full_influence(mesh: &Mesh, amount: f64) -> BTreeMap<u32, f64> {
        mesh.vertices
            .iter()
            .map(|vertex| (vertex.id, amount))
            .collect()
    }

    #[test]
    fn paint_creates_a_corner_layer_named_color_and_writes_once() {
        let mut mesh = Mesh::box_mesh(BoxParams::default()).unwrap();
        let corners: usize = mesh.faces.iter().map(|face| face.vertices.len()).sum();
        let influences = full_influence(&mesh, 1.0);
        let red = [1.0, 0.0, 0.0, 1.0];

        assert!(paint_vertex_colors(&mut mesh, &influences, red, "mix"));
        let store = color_layers::read(&mesh);
        assert_eq!(store.render.as_deref(), Some("color"));
        let layer = &store.layers[store.index_of("color").unwrap()];
        assert_eq!(layer.domain, ColorDomain::Corner);
        assert_eq!(layer.values.len(), corners);
        assert!(layer.values.iter().all(|value| *value == red));
        assert!(color_layers::validate(&mesh).is_ok());
        assert_eq!(
            mesh.attributes["color"]["domain"],
            serde_json::json!("corner")
        );

        // Painting the same color again changes nothing, so the store is not rewritten.
        assert!(!paint_vertex_colors(&mut mesh, &influences, red, "mix"));
    }

    #[test]
    fn paint_takes_a_point_layer_per_vertex() {
        let mut mesh = Mesh::box_mesh(BoxParams::default()).unwrap();
        let first_vertex = mesh.vertices[0].id;
        let mut store = color_layers::read(&mesh);
        store.layers.push(ColorLayer {
            name: "color".to_owned(),
            domain: ColorDomain::Point,
            values: vec![DEFAULT_COLOR; mesh.vertices.len()],
        });
        color_layers::write(&mut mesh, &store);
        let influences = BTreeMap::from([(first_vertex, 0.5)]);

        assert!(paint_vertex_colors(
            &mut mesh,
            &influences,
            [1.0, 1.0, 1.0, 1.0],
            "mix"
        ));
        let store = color_layers::read(&mesh);
        let layer = &store.layers[store.index_of("color").unwrap()];
        assert_eq!(layer.domain, ColorDomain::Point);
        assert_eq!(layer.values[0], [0.5, 0.5, 0.5, 1.0]);
        assert!(
            layer.values[1..]
                .iter()
                .all(|value| *value == DEFAULT_COLOR)
        );
    }
}
