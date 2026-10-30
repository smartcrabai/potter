use std::collections::BTreeSet;

use super::EvaluatorWorkspace;
use glam::DVec3;
use serde_json::{Value as JsonValue, json};

use super::texture_procedural_kernels::{
    self as procedural_texture, WaveBandsDirection, WaveProfile, WaveRingsDirection, WaveType,
};
#[cfg(test)]
use crate::image::sample_with_extension_alpha;
use crate::{
    error::{ErrorCode, PotError, Result},
    graph::{GraphKind, GraphNode, NodeGroup},
    image::{ImageData, ImageExtension},
    model::{Id, Registry, World},
};

#[derive(Clone, Copy)]
enum WorldValue {
    Scalar(f64),
    Color([f64; 4]),
    Vector(DVec3),
}

#[derive(Clone, Copy)]
enum WorldMixMode {
    Mix,
    Add,
    Multiply,
    Subtract,
    Divide,
}

#[derive(Clone, Copy)]
enum WorldMixError<'a> {
    UnsupportedDataType(&'a str),
    UnsupportedBlendType(&'a str),
}

fn world_mix_mode<'a>(
    data_type: Option<&'a str>,
    blend_type: Option<&'a str>,
) -> std::result::Result<WorldMixMode, WorldMixError<'a>> {
    let data_type = data_type.unwrap_or("FLOAT");
    if data_type != "RGBA" {
        return Err(WorldMixError::UnsupportedDataType(data_type));
    }

    match blend_type.unwrap_or("MIX") {
        "MIX" => Ok(WorldMixMode::Mix),
        "ADD" => Ok(WorldMixMode::Add),
        "MULTIPLY" => Ok(WorldMixMode::Multiply),
        "SUBTRACT" => Ok(WorldMixMode::Subtract),
        "DIVIDE" => Ok(WorldMixMode::Divide),
        unsupported => Err(WorldMixError::UnsupportedBlendType(unsupported)),
    }
}

pub(crate) fn world_mix_supported(data_type: Option<&str>, blend_type: Option<&str>) -> bool {
    world_mix_mode(data_type, blend_type).is_ok()
}

pub(crate) struct PreparedWorld {
    graph_id: Id,
    fold_index: super::inline_fold::FoldIndex,
}

impl PreparedWorld {
    pub(super) fn new(graph_id: &Id, group: &NodeGroup, node_groups: &Registry<NodeGroup>) -> Self {
        Self {
            graph_id: graph_id.clone(),
            fold_index: super::inline_fold::FoldIndex::new(graph_id, group, node_groups),
        }
    }
    pub(super) fn graph_id(&self) -> &Id {
        &self.graph_id
    }
}

pub(crate) fn prepare_world(
    world: &World,
    groups: &Registry<NodeGroup>,
) -> Result<Option<PreparedWorld>> {
    let Some((graph_id, group)) = world_graph(world, groups)? else {
        return Ok(None);
    };
    Ok(Some(PreparedWorld::new(graph_id, group, groups)))
}

/// Evaluates the World node graph for a single outgoing direction.
///
/// `path_traced` selects Cycles semantics when true and EEVEE semantics otherwise.
pub fn evaluate_world(
    world: &World,
    groups: &Registry<NodeGroup>,
    images: &std::collections::BTreeMap<String, ImageData>,
    direction: DVec3,
    path_traced: bool,
) -> Result<DVec3> {
    let mut workspace = EvaluatorWorkspace::default();
    evaluate_world_with_workspace(
        world,
        groups,
        images,
        direction,
        path_traced,
        &mut workspace,
    )
}

pub(crate) fn evaluate_world_with_workspace(
    world: &World,
    groups: &Registry<NodeGroup>,
    images: &std::collections::BTreeMap<String, ImageData>,
    direction: DVec3,
    path_traced: bool,
    workspace: &mut EvaluatorWorkspace,
) -> Result<DVec3> {
    let Some((graph_id, group)) = world_graph(world, groups)? else {
        return Ok(world_color(world));
    };
    let prepared = workspace.prepared_world(graph_id, group, groups);
    evaluate_world_graph(group, &prepared.fold_index, images, direction, path_traced)
}

pub(crate) fn evaluate_world_with_prepared(
    world: &World,
    groups: &Registry<NodeGroup>,
    images: &std::collections::BTreeMap<String, ImageData>,
    direction: DVec3,
    path_traced: bool,
    prepared: Option<&PreparedWorld>,
) -> Result<DVec3> {
    let Some((graph_id, group)) = world_graph(world, groups)? else {
        return Ok(world_color(world));
    };
    let prepared = prepared
        .filter(|prepared| prepared.graph_id == *graph_id)
        .ok_or_else(|| {
            PotError::new(
                ErrorCode::InternalError,
                "prepared World shader graph does not match the active world",
            )
        })?;
    evaluate_world_graph(group, &prepared.fold_index, images, direction, path_traced)
}

fn world_graph<'world, 'groups>(
    world: &'world World,
    groups: &'groups Registry<NodeGroup>,
) -> Result<Option<(&'world Id, &'groups NodeGroup)>> {
    let Some(graph_id) = &world.node_tree else {
        return Ok(None);
    };
    let group = groups.get(graph_id).ok_or_else(|| {
        PotError::with_details(
            ErrorCode::TargetNotFound,
            format!("world shader graph `{graph_id}` was not found"),
            json!({"graph_id": graph_id}),
        )
    })?;
    if group.kind != GraphKind::Shader {
        return Err(PotError::new(
            ErrorCode::InvalidOperation,
            format!("world graph `{graph_id}` is not a shader graph"),
        ));
    }
    Ok(Some((graph_id, group)))
}

fn world_color(world: &World) -> DVec3 {
    DVec3::from_array(world.color) * world.strength
}

fn evaluate_world_graph(
    group: &NodeGroup,
    fold_index: &super::inline_fold::FoldIndex,
    images: &std::collections::BTreeMap<String, ImageData>,
    direction: DVec3,
    path_traced: bool,
) -> Result<DVec3> {
    let output = group
        .nodes
        .iter()
        .find(|(_, node)| {
            matches!(
                node.node_type.as_str(),
                "ShaderNodeOutputWorld" | "OutputWorld"
            ) && node
                .properties
                .get("is_active_output")
                .and_then(JsonValue::as_bool)
                == Some(true)
        })
        .or_else(|| {
            group.nodes.iter().find(|(_, node)| {
                matches!(
                    node.node_type.as_str(),
                    "ShaderNodeOutputWorld" | "OutputWorld"
                )
            })
        })
        .map(|(id, _)| id)
        .ok_or_else(|| {
            PotError::new(
                ErrorCode::EvaluationFailed,
                "world graph has no OutputWorld node",
            )
        })?;
    let mut evaluator = WorldEvaluator {
        group,
        fold_index,
        images,
        direction: normalize_or(direction, DVec3::Z),
        path_traced,
        active: BTreeSet::new(),
    };
    let value = evaluator.input(output, "Surface", WorldValue::Color([0.0; 4]))?;
    let color = as_color(value);
    let color = DVec3::new(color[0], color[1], color[2]);
    if !color.is_finite() {
        return Err(PotError::new(
            ErrorCode::EvaluationFailed,
            "world graph produced a non-finite color",
        ));
    }
    Ok(color.max(DVec3::ZERO))
}

struct WorldEvaluator<'a> {
    group: &'a NodeGroup,
    fold_index: &'a super::inline_fold::FoldIndex,
    images: &'a std::collections::BTreeMap<String, ImageData>,
    direction: DVec3,
    path_traced: bool,
    active: BTreeSet<(Id, String)>,
}

impl<'a> WorldEvaluator<'a> {
    fn input(&mut self, id: &Id, socket: &str, fallback: WorldValue) -> Result<WorldValue> {
        if let Some(link) = self
            .group
            .links
            .iter()
            .find(|link| link.to_node == *id && socket_matches(&link.to_socket, socket))
        {
            return self.output(&link.from_node, &link.from_socket);
        }
        let node = Self::node(self.group, id)?;
        let raw = node
            .inputs
            .iter()
            .find(|(name, _)| socket_matches(name, socket))
            .map(|(_, value)| value);
        Ok(raw.map_or(fallback, |value| parse_value(value, fallback)))
    }

    fn output(&mut self, id: &Id, socket: &str) -> Result<WorldValue> {
        let key = (id.clone(), socket.to_owned());
        if !self.active.insert(key.clone()) {
            return Err(PotError::new(
                ErrorCode::EvaluationFailed,
                format!("world graph cycle at node `{id}`"),
            ));
        }
        let group = self.group;
        let node = Self::node(group, id)?;
        let result = self.eval_node(id, node, socket);
        self.active.remove(&key);
        result
    }
    fn node(group: &'a NodeGroup, id: &Id) -> Result<&'a GraphNode> {
        group.nodes.get(id).ok_or_else(|| {
            PotError::new(
                ErrorCode::InvalidOperation,
                format!("world graph references missing node `{id}`"),
            )
        })
    }
    fn procedural_texture_node(
        &mut self,
        id: &Id,
        node: &GraphNode,
        socket: &str,
    ) -> Result<WorldValue> {
        self.validate_procedural_texture_sockets(id, node)?;
        if node.node_type != "ShaderNodeTexWhiteNoise" {
            crate::shader::texture::validate_texture_mapping_only(
                id,
                node.node_type.as_str(),
                &node.properties,
            )?;
        }
        match node.node_type.as_str() {
            "ShaderNodeTexWave" => {
                if !matches!(socket, "Color" | "Fac") {
                    return Err(world_error(
                        id,
                        &format!("unknown Wave Texture output `{socket}`"),
                    ));
                }
                let wave_type = match world_texture_string(id, node, "wave_type", "BANDS")? {
                    "BANDS" => WaveType::Bands,
                    "RINGS" => WaveType::Rings,
                    value => {
                        return Err(world_texture_mode_error(
                            id,
                            "ShaderNodeTexWave",
                            "wave_type",
                            value,
                        ));
                    }
                };
                let bands_direction = if matches!(wave_type, WaveType::Bands) {
                    match world_texture_string(id, node, "bands_direction", "X")? {
                        "X" => WaveBandsDirection::X,
                        "Y" => WaveBandsDirection::Y,
                        "Z" => WaveBandsDirection::Z,
                        "DIAGONAL" => WaveBandsDirection::Diagonal,
                        value => {
                            return Err(world_texture_mode_error(
                                id,
                                "ShaderNodeTexWave",
                                "bands_direction",
                                value,
                            ));
                        }
                    }
                } else {
                    WaveBandsDirection::X
                };
                let rings_direction = if matches!(wave_type, WaveType::Rings) {
                    match world_texture_string(id, node, "rings_direction", "X")? {
                        "X" => WaveRingsDirection::X,
                        "Y" => WaveRingsDirection::Y,
                        "Z" => WaveRingsDirection::Z,
                        "SPHERICAL" => WaveRingsDirection::Spherical,
                        value => {
                            return Err(world_texture_mode_error(
                                id,
                                "ShaderNodeTexWave",
                                "rings_direction",
                                value,
                            ));
                        }
                    }
                } else {
                    WaveRingsDirection::X
                };
                let profile = match world_texture_string(id, node, "wave_profile", "SIN")? {
                    "SIN" => WaveProfile::Sine,
                    "SAW" => WaveProfile::Saw,
                    "TRI" => WaveProfile::Triangle,
                    value => {
                        return Err(world_texture_mode_error(
                            id,
                            "ShaderNodeTexWave",
                            "wave_profile",
                            value,
                        ));
                    }
                };
                let value = procedural_texture::wave(
                    self.procedural_texture_vector(id, self.direction)?,
                    self.procedural_texture_scalar(id, "Scale", 5.0)?,
                    self.procedural_texture_scalar(id, "Distortion", 0.0)?,
                    self.procedural_texture_scalar(id, "Detail", 2.0)?,
                    self.procedural_texture_scalar(id, "Detail Scale", 1.0)?,
                    self.procedural_texture_scalar(id, "Detail Roughness", 0.5)?,
                    self.procedural_texture_scalar(id, "Phase Offset", 0.0)?,
                    wave_type,
                    bands_direction,
                    rings_direction,
                    profile,
                );
                match socket {
                    "Color" => Ok(WorldValue::Color([
                        f64::from(value),
                        f64::from(value),
                        f64::from(value),
                        1.0,
                    ])),
                    "Fac" => Ok(WorldValue::Scalar(f64::from(value))),
                    _ => Err(world_error(id, "unknown Wave Texture output")),
                }
            }
            "ShaderNodeTexMagic" => {
                if !matches!(socket, "Color" | "Fac") {
                    return Err(world_error(
                        id,
                        &format!("unknown Magic Texture output `{socket}`"),
                    ));
                }
                let depth =
                    world_texture_integer(id, node, "turbulence_depth", 2)?.clamp(0, 10) as u8;
                let vector = self.procedural_texture_vector(id, self.direction)?;
                let scale = self.procedural_texture_scalar(id, "Scale", 5.0)?;
                let distortion = self.procedural_texture_scalar(id, "Distortion", 1.0)?;
                let (color, factor) = if self.fold_index.primitive_nodes.contains(id) {
                    procedural_texture::magic(vector, scale, distortion, depth)
                } else {
                    procedural_texture::magic_shader(
                        vector,
                        scale,
                        distortion,
                        depth,
                        self.path_traced,
                    )
                };
                match socket {
                    "Color" => Ok(WorldValue::Color(color.map(f64::from))),
                    "Fac" => Ok(WorldValue::Scalar(f64::from(factor))),
                    _ => Err(world_error(id, "unknown Magic Texture output")),
                }
            }
            "ShaderNodeTexBrick" => {
                if !matches!(socket, "Color" | "Fac") {
                    return Err(world_error(
                        id,
                        &format!("unknown Brick Texture output `{socket}`"),
                    ));
                }
                let offset = world_texture_float(id, node, "offset", 0.5)?;
                let offset_frequency = world_texture_integer(id, node, "offset_frequency", 2)?;
                let squash = world_texture_float(id, node, "squash", 1.0)?;
                let squash_frequency = world_texture_integer(id, node, "squash_frequency", 2)?;
                let vector = self.procedural_texture_vector(id, self.direction)?;
                let color1 = self.procedural_texture_color(id, "Color1", [0.8, 0.8, 0.8, 1.0])?;
                let color2 = self.procedural_texture_color(id, "Color2", [0.2, 0.2, 0.2, 1.0])?;
                let mortar = self.procedural_texture_color(id, "Mortar", [0.0, 0.0, 0.0, 1.0])?;
                let scale = self.procedural_texture_scalar(id, "Scale", 5.0)?;
                let mortar_size = self.procedural_texture_scalar(id, "Mortar Size", 0.02)?;
                let mortar_smooth = self.procedural_texture_scalar(id, "Mortar Smooth", 0.1)?;
                let bias = self.procedural_texture_scalar(id, "Bias", 0.0)?;
                let width = self.procedural_texture_scalar(id, "Brick Width", 0.5)?;
                let row_height = self.procedural_texture_scalar(id, "Row Height", 0.25)?;
                if width == 0.0 || row_height == 0.0 {
                    return Err(PotError::new(
                        ErrorCode::EvaluationFailed,
                        "Brick Texture width and row height must be nonzero",
                    ));
                }
                let (mut color, factor) = procedural_texture::brick(
                    vector,
                    color1,
                    color2,
                    mortar,
                    scale,
                    mortar_size,
                    mortar_smooth,
                    bias,
                    width,
                    row_height,
                    offset,
                    offset_frequency,
                    squash,
                    squash_frequency,
                );
                if self.path_traced {
                    // Cycles SVM stores Brick colors as RGB; EEVEE preserves the socket alpha.
                    color[3] = 1.0;
                }
                match socket {
                    "Color" => Ok(WorldValue::Color(color.map(f64::from))),
                    "Fac" => Ok(WorldValue::Scalar(f64::from(factor))),
                    _ => Err(world_error(id, "unknown Brick Texture output")),
                }
            }
            "ShaderNodeTexWhiteNoise" => {
                if !matches!(socket, "Value" | "Color") {
                    return Err(world_error(
                        id,
                        &format!("unknown White Noise output `{socket}`"),
                    ));
                }
                let dimensions = match world_texture_string(id, node, "noise_dimensions", "3D")? {
                    "1D" => 1,
                    "2D" => 2,
                    "3D" => 3,
                    "4D" => 4,
                    value => {
                        return Err(world_texture_mode_error(
                            id,
                            "ShaderNodeTexWhiteNoise",
                            "noise_dimensions",
                            value,
                        ));
                    }
                };
                let vector = if dimensions == 1 {
                    [0.0; 3]
                } else {
                    self.procedural_texture_vector_input(id, "Vector", [0.0; 3])?
                };
                let w = if matches!(dimensions, 1 | 4) {
                    self.procedural_texture_scalar(id, "W", 0.0)?
                } else {
                    0.0
                };
                match socket {
                    "Value" => procedural_texture::white_noise_value(dimensions, vector, w)
                        .map(|value| WorldValue::Scalar(f64::from(value)))
                        .ok_or_else(|| world_error(id, "White Noise dimensions are invalid")),
                    "Color" => procedural_texture::white_noise_color(dimensions, vector, w)
                        .map(|color| WorldValue::Color(color.map(f64::from)))
                        .ok_or_else(|| world_error(id, "White Noise dimensions are invalid")),
                    _ => Err(world_error(id, "unknown White Noise output")),
                }
            }
            "ShaderNodeTexGabor" => {
                if !matches!(socket, "Value" | "Phase" | "Intensity") {
                    return Err(world_error(
                        id,
                        &format!("unknown Gabor Texture output `{socket}`"),
                    ));
                }
                let dimensions = match world_texture_string(id, node, "gabor_type", "2D")? {
                    "2D" => 2,
                    "3D" => 3,
                    value => {
                        return Err(world_texture_mode_error(
                            id,
                            "ShaderNodeTexGabor",
                            "gabor_type",
                            value,
                        ));
                    }
                };
                let vector = self.procedural_texture_vector(id, self.direction)?;
                let scale = self.procedural_texture_scalar(id, "Scale", 5.0)?;
                let frequency = self.procedural_texture_scalar(id, "Frequency", 2.0)?;
                let anisotropy = self.procedural_texture_scalar(id, "Anisotropy", 1.0)?;
                let values = if dimensions == 2 {
                    procedural_texture::gabor_2d(
                        vector,
                        scale,
                        frequency,
                        anisotropy,
                        self.procedural_texture_scalar(
                            id,
                            "Orientation 2D",
                            std::f32::consts::FRAC_PI_4,
                        )?,
                    )
                } else {
                    let orientation = self.procedural_texture_vector_input(
                        id,
                        "Orientation 3D",
                        [std::f32::consts::SQRT_2, std::f32::consts::SQRT_2, 0.0],
                    )?;
                    if self.fold_index.primitive_nodes.contains(id) {
                        procedural_texture::gabor_3d(
                            vector,
                            scale,
                            frequency,
                            anisotropy,
                            orientation,
                        )
                    } else {
                        procedural_texture::gabor_3d_shader(
                            vector,
                            scale,
                            frequency,
                            anisotropy,
                            orientation,
                            self.path_traced,
                        )
                    }
                };
                let value = match socket {
                    "Value" => values[0],
                    "Phase" => values[1],
                    "Intensity" => values[2],
                    _ => return Err(world_error(id, "unknown Gabor Texture output")),
                };
                Ok(WorldValue::Scalar(f64::from(value)))
            }
            _ => Err(world_error(id, "unknown procedural texture node")),
        }
    }

    fn validate_procedural_texture_sockets(&self, id: &Id, node: &GraphNode) -> Result<()> {
        let valid: &[&str] = match node.node_type.as_str() {
            "ShaderNodeTexWave" => &[
                "Vector",
                "Scale",
                "Distortion",
                "Detail",
                "Detail Scale",
                "Detail Roughness",
                "Phase Offset",
            ],
            "ShaderNodeTexMagic" => &["Vector", "Scale", "Distortion"],
            "ShaderNodeTexBrick" => &[
                "Vector",
                "Color1",
                "Color2",
                "Mortar",
                "Scale",
                "Mortar Size",
                "Mortar Smooth",
                "Bias",
                "Brick Width",
                "Row Height",
            ],
            "ShaderNodeTexWhiteNoise" => &["Vector", "W"],
            "ShaderNodeTexGabor" => &[
                "Vector",
                "Scale",
                "Frequency",
                "Anisotropy",
                "Orientation 2D",
                "Orientation 3D",
            ],
            _ => return Err(world_error(id, "unknown procedural texture node")),
        };
        if let Some(socket) = node
            .inputs
            .keys()
            .find(|socket| !valid.contains(&socket.as_str()))
        {
            return Err(world_error(
                id,
                &format!("unknown procedural texture input `{socket}`"),
            ));
        }
        let inactive = match node.node_type.as_str() {
            "ShaderNodeTexWhiteNoise" => {
                match world_texture_string(id, node, "noise_dimensions", "3D")? {
                    "1D" => Some("Vector"),
                    "2D" | "3D" => Some("W"),
                    _ => None,
                }
            }
            "ShaderNodeTexGabor" => match world_texture_string(id, node, "gabor_type", "2D")? {
                "2D" => Some("Orientation 3D"),
                "3D" => Some("Orientation 2D"),
                _ => None,
            },
            _ => None,
        };
        if let Some(socket) = inactive
            && self
                .group
                .links
                .iter()
                .any(|link| link.to_node == *id && link.to_socket == socket)
        {
            return Err(world_error(
                id,
                &format!("procedural texture input `{socket}` is unavailable"),
            ));
        }
        Ok(())
    }

    fn procedural_texture_vector(&mut self, id: &Id, default: DVec3) -> Result<[f32; 3]> {
        let linked = self
            .group
            .links
            .iter()
            .any(|link| link.to_node == *id && socket_matches(&link.to_socket, "Vector"));
        let value = if linked {
            self.input(id, "Vector", WorldValue::Vector(default))?
        } else {
            WorldValue::Vector(default)
        };
        world_texture_vector(as_vector(value))
    }

    fn procedural_texture_vector_input(
        &mut self,
        id: &Id,
        socket: &str,
        default: [f32; 3],
    ) -> Result<[f32; 3]> {
        let fallback = WorldValue::Vector(DVec3::new(
            f64::from(default[0]),
            f64::from(default[1]),
            f64::from(default[2]),
        ));
        world_texture_vector(as_vector(self.input(id, socket, fallback)?))
    }

    fn procedural_texture_scalar(&mut self, id: &Id, socket: &str, default: f32) -> Result<f32> {
        world_texture_value(as_scalar(self.input(
            id,
            socket,
            WorldValue::Scalar(f64::from(default)),
        )?))
    }

    fn procedural_texture_color(
        &mut self,
        id: &Id,
        socket: &str,
        default: [f32; 4],
    ) -> Result<[f32; 4]> {
        let fallback = WorldValue::Color(default.map(f64::from));
        world_texture_color(as_color(self.input(id, socket, fallback)?))
    }

    fn eval_node(&mut self, id: &Id, node: &GraphNode, socket: &str) -> Result<WorldValue> {
        match node.node_type.as_str() {
            "OutputWorld" | "ShaderNodeOutputWorld" => Err(PotError::new(
                ErrorCode::InvalidOperation,
                "world output node has no outputs",
            )),
            "ShaderNodeCombineXYZ" | "CombineXYZ" => {
                if socket != "Vector" {
                    return Err(world_error(
                        id,
                        &format!("unknown Combine XYZ output `{socket}`"),
                    ));
                }
                let x = as_scalar(self.input(id, "X", WorldValue::Scalar(0.0))?);
                let y = as_scalar(self.input(id, "Y", WorldValue::Scalar(0.0))?);
                let z = as_scalar(self.input(id, "Z", WorldValue::Scalar(0.0))?);
                Ok(WorldValue::Vector(DVec3::new(x, y, z)))
            }
            "ShaderNodeBackground" => {
                let color = as_color(self.input(
                    id,
                    "Color",
                    WorldValue::Color([0.05, 0.05, 0.05, 1.0]),
                )?);
                let strength = as_scalar(self.input(id, "Strength", WorldValue::Scalar(1.0))?);
                Ok(WorldValue::Color([
                    color[0] * strength,
                    color[1] * strength,
                    color[2] * strength,
                    color[3],
                ]))
            }
            "ShaderNodeEmission" => {
                let color =
                    as_color(self.input(id, "Color", WorldValue::Color([1.0, 1.0, 1.0, 1.0]))?);
                let strength = as_scalar(self.input(id, "Strength", WorldValue::Scalar(1.0))?);
                Ok(WorldValue::Color([
                    color[0] * strength,
                    color[1] * strength,
                    color[2] * strength,
                    color[3],
                ]))
            }
            "ShaderNodeTexSky" => {
                if node
                    .properties
                    .get("sky_type")
                    .is_none_or(|value| sky_property_is_default("sky_type", value))
                {
                    return Err(unsupported_sky_feature(id, "multiple_scattering_model"));
                }
                crate::shader::texture::validate_texture_node_mapping(
                    id,
                    &node.node_type,
                    &node.properties,
                )?;
                for (property, value) in &node.properties {
                    if matches!(
                        property.as_str(),
                        "sky_type"
                            | "sun_disc"
                            | "sun_size"
                            | "sun_intensity"
                            | "air_density"
                            | "aerosol_density"
                            | "ozone_density"
                            | "sun_elevation"
                            | "sun_rotation"
                            | "altitude"
                            | "turbidity"
                            | "ground_albedo"
                    ) && !sky_property_is_default(property, value)
                    {
                        return Err(unsupported_sky_feature(id, property));
                    }
                }
                if let Some(input) = node.inputs.keys().find(|input| {
                    !socket_matches(input, "Sun Direction") && !socket_matches(input, "Vector")
                }) {
                    return Err(unsupported_sky_feature(id, &format!("input.{input}")));
                }
                if let Some(link) = self.group.links.iter().find(|link| {
                    link.to_node == *id
                        && !socket_matches(&link.to_socket, "Sun Direction")
                        && !socket_matches(&link.to_socket, "Vector")
                }) {
                    return Err(unsupported_sky_feature(
                        id,
                        &format!("input.{}", link.to_socket),
                    ));
                }
                let sun = normalize_or(
                    as_vector(self.input(id, "Sun Direction", WorldValue::Vector(DVec3::Z))?),
                    DVec3::Z,
                );
                let direction = normalize_or(
                    as_vector(self.input(id, "Vector", WorldValue::Vector(self.direction))?),
                    self.direction,
                );
                Ok(WorldValue::Color(sky_radiance(direction, sun)))
            }
            "ShaderNodeTexEnvironment" => {
                crate::shader::texture::validate_texture_node_mapping(
                    id,
                    &node.node_type,
                    &node.properties,
                )?;
                let direction =
                    if self.group.links.iter().any(|link| {
                        link.to_node == *id && socket_matches(&link.to_socket, "Vector")
                    }) {
                        as_vector(self.input(id, "Vector", WorldValue::Vector(self.direction))?)
                    } else {
                        self.direction
                    };
                let image_id = node
                    .properties
                    .get("image")
                    .or_else(|| node.properties.get("image_id"))
                    .and_then(JsonValue::as_str)
                    .ok_or_else(|| world_error(id, "Environment Texture requires an image ID"))?;
                let image = self.images.get(image_id).ok_or_else(|| {
                    PotError::with_details(
                        ErrorCode::TargetNotFound,
                        format!("environment image `{image_id}` was not found"),
                        json!({"node_id": id, "image_id": image_id}),
                    )
                })?;
                let interpolation = crate::shader::texture::image_texture_interpolation(
                    id,
                    &node.node_type,
                    &node.properties,
                    image.interpolation,
                )?;
                if !self.path_traced
                    && interpolation == crate::image::ImageInterpolation::Linear
                    && node
                        .properties
                        .get("projection")
                        .and_then(JsonValue::as_str)
                        .is_some_and(|projection| projection.eq_ignore_ascii_case("MIRROR_BALL"))
                {
                    return Err(
                        crate::shader::texture::unsupported_realtime_environment_filtering(
                            id, image_id,
                        ),
                    );
                }
                let uv =
                    environment_texture_uv(id, node, direction, self.direction, self.path_traced)?;
                let extensions = if self.path_traced {
                    [ImageExtension::Repeat; 2]
                } else if node
                    .properties
                    .get("projection")
                    .and_then(JsonValue::as_str)
                    .is_some_and(|projection| projection.eq_ignore_ascii_case("MIRROR_BALL"))
                {
                    [ImageExtension::Extend; 2]
                } else {
                    [ImageExtension::Repeat, ImageExtension::Extend]
                };
                let color = crate::shader::texture::sample_environment_texture(
                    image,
                    uv,
                    interpolation,
                    extensions,
                    self.path_traced,
                );
                Ok(WorldValue::Color(color))
            }
            "ShaderNodeTexWave"
            | "ShaderNodeTexMagic"
            | "ShaderNodeTexBrick"
            | "ShaderNodeTexWhiteNoise"
            | "ShaderNodeTexGabor" => self.procedural_texture_node(id, node, socket),
            "ShaderNodeTexCoord" => match socket {
                "Normal" | "Generated" | "Object" | "Window" | "Camera" | "Incoming" => {
                    Ok(WorldValue::Vector(self.direction))
                }
                _ => Err(world_error(
                    id,
                    &format!("unknown Texture Coordinate output `{socket}`"),
                )),
            },
            "ShaderNodeMixRGB" => {
                let factor =
                    as_scalar(self.input(id, "Fac", WorldValue::Scalar(0.5))?).clamp(0.0, 1.0);
                let first =
                    as_color(self.input(id, "Color1", WorldValue::Color([0.0, 0.0, 0.0, 1.0]))?);
                let second =
                    as_color(self.input(id, "Color2", WorldValue::Color([1.0, 1.0, 1.0, 1.0]))?);
                let result = std::array::from_fn(|index| {
                    first[index] + (second[index] - first[index]) * factor
                });
                Ok(WorldValue::Color(result))
            }
            "ShaderNodeMix" => {
                let mode = world_mix_mode(
                    node.properties
                        .get("data_type")
                        .and_then(JsonValue::as_str),
                    node.properties
                        .get("blend_type")
                        .and_then(JsonValue::as_str),
                )
                .map_err(|error| match error {
                    WorldMixError::UnsupportedDataType(data_type) => PotError::with_details(
                        ErrorCode::UnsupportedFeature,
                        format!("world ShaderNodeMix data type `{data_type}` is not supported"),
                        json!({
                            "feature_id": format!("world.node.ShaderNodeMix.data_type.{data_type}"),
                            "node_id": id,
                        }),
                    ),
                    WorldMixError::UnsupportedBlendType(unsupported) => {
                        PotError::with_details(
                            ErrorCode::UnsupportedFeature,
                            format!(
                                "world ShaderNodeMix blend type `{unsupported}` is not supported"
                            ),
                            json!({
                                "feature_id": format!("world.node.ShaderNodeMix.blend_type.{unsupported}"),
                                "node_id": id,
                            }),
                        )
                    }
                })?;
                let factor = as_scalar(self.input(id, "Factor_Float", WorldValue::Scalar(0.5))?)
                    .clamp(0.0, 1.0);
                let first =
                    as_color(self.input(id, "A_Color", WorldValue::Color([0.5, 0.5, 0.5, 1.0]))?);
                let second =
                    as_color(self.input(id, "B_Color", WorldValue::Color([0.5, 0.5, 0.5, 1.0]))?);
                let blended = match mode {
                    WorldMixMode::Mix => second,
                    WorldMixMode::Add => std::array::from_fn(|index| first[index] + second[index]),
                    WorldMixMode::Multiply => {
                        std::array::from_fn(|index| first[index] * second[index])
                    }
                    WorldMixMode::Subtract => {
                        std::array::from_fn(|index| first[index] - second[index])
                    }
                    WorldMixMode::Divide => std::array::from_fn(|index| {
                        if second[index].abs() <= f64::EPSILON {
                            first[index]
                        } else {
                            first[index] / second[index]
                        }
                    }),
                };
                Ok(WorldValue::Color(std::array::from_fn(|index| {
                    first[index] + (blended[index] - first[index]) * factor
                })))
            }
            "ShaderNodeMixShader" => {
                let factor =
                    as_scalar(self.input(id, "Fac", WorldValue::Scalar(0.5))?).clamp(0.0, 1.0);
                let first = as_color(self.input(id, "Shader", WorldValue::Color([0.0; 4]))?);
                let second = as_color(self.input(id, "Shader_001", WorldValue::Color([0.0; 4]))?);
                Ok(WorldValue::Color(std::array::from_fn(|index| {
                    first[index] + (second[index] - first[index]) * factor
                })))
            }
            "ShaderNodeAddShader" => {
                let first = as_color(self.input(id, "Shader", WorldValue::Color([0.0; 4]))?);
                let second = as_color(self.input(id, "Shader_001", WorldValue::Color([0.0; 4]))?);
                Ok(WorldValue::Color(std::array::from_fn(|index| {
                    first[index] + second[index]
                })))
            }
            unknown => Err(PotError::with_details(
                ErrorCode::UnsupportedFeature,
                format!("world node `{unknown}` is not supported"),
                json!({"feature_id": format!("world.node.{unknown}"), "node_id": id}),
            )),
        }
    }
}
fn unsupported_sky_feature(id: &Id, feature: &str) -> PotError {
    PotError::with_details(
        ErrorCode::UnsupportedFeature,
        format!("world ShaderNodeTexSky setting `{feature}` is not supported"),
        json!({
            "feature_id":format!("world.node.ShaderNodeTexSky.{feature}"),
            "node_id":id,
        }),
    )
}

pub(crate) fn sky_property_is_default(property: &str, value: &JsonValue) -> bool {
    match property {
        "sky_type" => value.as_str() == Some("MULTIPLE_SCATTERING"),
        "sun_disc" => value.as_bool() == Some(true),
        "sun_size" => sky_float_is_default(value, 0.545_f32 * (std::f32::consts::PI / 180.0)),
        "sun_intensity" | "air_density" | "aerosol_density" | "ozone_density" => {
            sky_float_is_default(value, 1.0)
        }
        "sun_elevation" => sky_float_is_default(value, 15.0_f32 * (std::f32::consts::PI / 180.0)),
        "sun_rotation" => sky_float_is_default(value, 0.0),
        "altitude" => sky_float_is_default(value, 100.0),
        "turbidity" => sky_float_is_default(value, 2.2),
        "ground_albedo" => sky_float_is_default(value, 0.3),
        "texture_mapping" => sky_rna_marker(value, "TexMapping"),
        "color_mapping" => sky_rna_marker(value, "ColorMapping"),
        _ => false,
    }
}

fn sky_float_is_default(value: &JsonValue, default: f32) -> bool {
    let Some(value) = value.as_f64().map(|value| value as f32) else {
        return false;
    };
    let value_bits = value.to_bits();
    let default_bits = default.to_bits();
    value_bits == default_bits
        || (value_bits.trailing_zeros() >= 31 && default_bits.trailing_zeros() >= 31)
}

fn sky_rna_marker(value: &JsonValue, rna_type: &str) -> bool {
    value.as_object().is_some_and(|object| {
        object.len() == 1 && object.get("rna_type").and_then(JsonValue::as_str) == Some(rna_type)
    })
}

fn sky_radiance(direction: DVec3, sun: DVec3) -> [f64; 4] {
    let elevation = sun.z.clamp(-1.0, 1.0).asin();
    let daylight = ((elevation + 0.12) / 0.9).clamp(0.0, 1.0);
    let horizon = (-direction.z.abs() * 3.0).exp();
    let zenith = direction.z.max(0.0).sqrt();
    let sun_alignment = direction.dot(sun).clamp(0.0, 1.0);
    let disc = sun_alignment.powi(96) * (0.5 + daylight * 7.0);
    let warmth = (1.0 - daylight) * horizon;
    [
        0.025 + daylight * (0.12 + 0.36 * zenith) + warmth * 0.62 + disc,
        0.035 + daylight * (0.24 + 0.43 * zenith) + warmth * 0.20 + disc * 0.78,
        0.06 + daylight * (0.48 + 0.38 * zenith) + warmth * 0.06 + disc * 0.48,
        1.0,
    ]
}
fn world_texture_string<'a>(
    id: &Id,
    node: &'a GraphNode,
    property: &str,
    default: &'a str,
) -> Result<&'a str> {
    match node.properties.get(property) {
        None => Ok(default),
        Some(value) => value.as_str().ok_or_else(|| {
            world_error(
                id,
                &format!("texture property `{property}` must be a string"),
            )
        }),
    }
}

fn world_texture_float(id: &Id, node: &GraphNode, property: &str, default: f32) -> Result<f32> {
    let value = match node.properties.get(property) {
        None => f64::from(default),
        Some(value) => value
            .as_f64()
            .filter(|value| value.is_finite())
            .ok_or_else(|| world_error(id, &format!("texture property `{property}` is invalid")))?,
    } as f32;
    if value.is_finite() {
        Ok(value)
    } else {
        Err(world_error(
            id,
            &format!("texture property `{property}` exceeds float range"),
        ))
    }
}

fn world_texture_integer(id: &Id, node: &GraphNode, property: &str, default: i32) -> Result<i32> {
    match node.properties.get(property) {
        None => Ok(default),
        Some(value) => value
            .as_i64()
            .and_then(|value| i32::try_from(value).ok())
            .ok_or_else(|| world_error(id, &format!("texture property `{property}` is invalid"))),
    }
}

fn world_texture_value(value: f64) -> Result<f32> {
    let value = value as f32;
    if value.is_finite() {
        Ok(value)
    } else {
        Err(PotError::new(
            ErrorCode::EvaluationFailed,
            "world procedural texture input exceeds the float range",
        ))
    }
}

fn world_texture_vector(value: DVec3) -> Result<[f32; 3]> {
    let vector = [value.x as f32, value.y as f32, value.z as f32];
    if vector.iter().all(|component| component.is_finite()) {
        Ok(vector)
    } else {
        Err(PotError::new(
            ErrorCode::EvaluationFailed,
            "world procedural texture vector exceeds the float range",
        ))
    }
}

fn world_texture_color(value: [f64; 4]) -> Result<[f32; 4]> {
    let color = [
        value[0] as f32,
        value[1] as f32,
        value[2] as f32,
        value[3] as f32,
    ];
    if color.iter().all(|component| component.is_finite()) {
        Ok(color)
    } else {
        Err(PotError::new(
            ErrorCode::EvaluationFailed,
            "world procedural texture color exceeds the float range",
        ))
    }
}

fn world_texture_mode_error(id: &Id, node_type: &str, property: &str, mode: &str) -> PotError {
    PotError::with_details(
        ErrorCode::UnsupportedFeature,
        format!("world `{node_type}` mode `{mode}` is not implemented"),
        json!({
            "feature_id": format!("shader.node.{node_type}.{property}"),
            "node_id": id,
            "value": mode,
        }),
    )
}

fn parse_value(value: &JsonValue, fallback: WorldValue) -> WorldValue {
    match fallback {
        WorldValue::Scalar(default) => WorldValue::Scalar(value.as_f64().unwrap_or(default)),
        WorldValue::Color(default) => {
            let Some(values) = value.as_array() else {
                return WorldValue::Color(default);
            };
            let mut color = default;
            for (index, destination) in color.iter_mut().enumerate() {
                if let Some(number) = values.get(index).and_then(JsonValue::as_f64) {
                    *destination = number;
                }
            }
            WorldValue::Color(color)
        }
        WorldValue::Vector(default) => {
            let Some(values) = value.as_array() else {
                return WorldValue::Vector(default);
            };
            let mut vector = default;
            for index in 0..3 {
                if let Some(number) = values.get(index).and_then(JsonValue::as_f64) {
                    vector[index] = number;
                }
            }
            WorldValue::Vector(vector)
        }
    }
}

fn as_scalar(value: WorldValue) -> f64 {
    match value {
        WorldValue::Scalar(value) => value,
        WorldValue::Color(color) => color[0],
        WorldValue::Vector(vector) => vector.x,
    }
}

fn as_color(value: WorldValue) -> [f64; 4] {
    match value {
        WorldValue::Scalar(value) => [value, value, value, 1.0],
        WorldValue::Color(color) => color,
        WorldValue::Vector(vector) => [vector.x, vector.y, vector.z, 1.0],
    }
}

fn as_vector(value: WorldValue) -> DVec3 {
    match value {
        WorldValue::Scalar(value) => DVec3::splat(value),
        WorldValue::Color(color) => DVec3::new(color[0], color[1], color[2]),
        WorldValue::Vector(vector) => vector,
    }
}

fn socket_matches(candidate: &str, requested: &str) -> bool {
    candidate == requested
        || (requested == "Color" && candidate == "A")
        || (requested == "Color1" && candidate == "A")
        || (requested == "Color2" && candidate == "B")
        || (requested == "A_Color" && candidate == "A")
        || (requested == "B_Color" && candidate == "B")
        || (requested == "Factor_Float" && candidate == "Factor")
        || (requested == "Shader" && candidate == "Shader 1")
        || (requested == "Shader_001" && candidate == "Shader 2")
        || (requested == "Fac" && candidate == "Factor")
}

fn normalize_or(value: DVec3, fallback: DVec3) -> DVec3 {
    let normalized = value.normalize_or_zero();
    if normalized == DVec3::ZERO || !normalized.is_finite() {
        fallback
    } else {
        normalized
    }
}

fn world_error(id: &Id, message: &str) -> PotError {
    PotError::with_details(ErrorCode::InvalidOperation, message, json!({"node_id": id}))
}

/// The equirectangular coordinate of a world ray direction in Blender's V-up convention, as Cycles'
/// `direction_to_equirectangular` gives it (`intern/cycles/kernel/camera/projection.h`, range (-2π, π, -π, π)):
/// u = 0.5 − atan2(y, x) / 2π and v = 1 − acos(z) / π, so v = 1 is the zenith.
pub(crate) fn equirectangular_uv(direction: DVec3) -> [f64; 2] {
    let u = (0.5 - direction.y.atan2(direction.x) / std::f64::consts::TAU).rem_euclid(1.0);
    let v = 1.0 - direction.z.clamp(-1.0, 1.0).acos() / std::f64::consts::PI;
    [u, v]
}

#[cfg(test)]
fn environment_colour(image: &ImageData, direction: DVec3) -> [f64; 4] {
    sample_with_extension_alpha(
        image,
        equirectangular_uv(direction),
        false,
        image.interpolation,
        ImageExtension::Repeat,
        false,
    )
}
fn eevee_equirectangular_uv(direction: DVec3) -> [f64; 2] {
    let mut direction = direction.as_vec3();
    let length_squared = direction.length_squared();
    // Match Blender's safe_normalize, including its exact zero-component UV fallback.
    if length_squared > 1.0e-35 {
        direction /= length_squared.sqrt();
    } else {
        direction = glam::Vec3::X;
    }
    if direction.x == 0.0 || direction.y == 0.0 {
        return [0.0; 2];
    }
    [
        f64::from(-direction.y.atan2(direction.x) / (2.0 * std::f32::consts::PI) + 0.5),
        f64::from(direction.z.atan2(direction.x.hypot(direction.y)) / std::f32::consts::PI + 0.5),
    ]
}

pub(crate) fn environment_texture_uv(
    id: &Id,
    node: &GraphNode,
    direction: DVec3,
    fallback: DVec3,
    path_traced: bool,
) -> Result<[f64; 2]> {
    let equirectangular = |direction| {
        if path_traced {
            equirectangular_uv(normalize_or(direction, fallback))
        } else {
            eevee_equirectangular_uv(direction)
        }
    };
    match node.properties.get("projection") {
        None => Ok(equirectangular(direction)),
        Some(JsonValue::String(projection))
            if projection.eq_ignore_ascii_case("EQUIRECTANGULAR") =>
        {
            Ok(equirectangular(direction))
        }
        Some(JsonValue::String(projection)) if projection.eq_ignore_ascii_case("MIRROR_BALL") => {
            if path_traced {
                let mut direction = direction.normalize_or_zero();
                direction.y -= 1.0;
                let divisor = 2.0 * (-0.5 * direction.y).max(0.0).sqrt();
                if divisor > 0.0 {
                    direction /= divisor;
                }
                Ok([
                    f64::midpoint(direction.x, 1.0),
                    f64::midpoint(direction.z, 1.0),
                ])
            } else {
                let vector =
                    glam::Vec3::new(direction.x as f32, direction.y as f32, direction.z as f32);
                // EEVEE's vector_math_safe_normalize uses this squared-length cutoff.
                let mut direction = if vector.dot(vector) > 1.0e-35_f32 {
                    vector.normalize()
                } else {
                    glam::Vec3::ZERO
                };
                direction.y -= 1.0;
                let divisor = 2.0 * (-0.5 * direction.y).max(0.0).sqrt();
                direction /= divisor.max(1.0e-8);
                Ok([
                    f64::from(0.5 * direction.x + 0.5),
                    f64::from(0.5 * direction.z + 0.5),
                ])
            }
        }
        Some(projection) => Err(PotError::with_details(
            ErrorCode::UnsupportedFeature,
            format!("Environment Texture projection `{projection}` is not supported"),
            json!({
                "feature_id": "shader.node.ShaderNodeTexEnvironment.projection",
                "node_id": id,
                "projection": projection,
            }),
        )),
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::graph::{GraphKind, GraphLink};

    #[test]
    fn eevee_environment_uv_uses_only_exact_zero_fallback() {
        assert_eq!(
            eevee_equirectangular_uv(DVec3::new(0.0, 1.0, 0.0)),
            [0.0; 2]
        );
        assert_eq!(
            eevee_equirectangular_uv(DVec3::new(1.0, 0.0, 0.0)),
            [0.0; 2]
        );

        let tiny = f64::from(f32::MIN_POSITIVE);
        assert_ne!(
            eevee_equirectangular_uv(DVec3::new(tiny, 1.0, 0.0)),
            [0.0; 2]
        );
    }
    #[test]
    fn mirror_ball_uses_eevee_safe_normalize_cutoff() -> Result<()> {
        let mut node = GraphNode::new("ShaderNodeTexEnvironment");
        node.properties
            .insert("projection".to_owned(), json!("MIRROR_BALL"));
        let id = Id::new("mirror_ball")?;
        let below =
            environment_texture_uv(&id, &node, DVec3::new(1.0e-18, 0.0, 0.0), DVec3::Z, false)?;
        let above =
            environment_texture_uv(&id, &node, DVec3::new(1.0e-17, 0.0, 0.0), DVec3::Z, false)?;

        assert!((below[0] - 0.5).abs() <= f64::EPSILON);
        assert!((below[1] - 0.5).abs() <= f64::EPSILON);
        assert!((above[0] - 0.853_553_390_593_273_7).abs() <= 1.0e-7);
        assert!((above[1] - 0.5).abs() <= f64::EPSILON);
        Ok(())
    }

    /// Blender 5.2.2's environment colours (`tests/fixtures/repeat_edge/generate_world_env_oracle.py`): 256 ray directions
    /// around the u = 0 seam and up to the zenith, each with the colour of a 4 x 2 equirectangular texture, Closest and
    /// Linear. The fixture rounds colours to six decimals, so the tolerance is 1e-6.
    #[test]
    fn environment_samples_match_blenders_render() -> Result<(), Box<dyn std::error::Error>> {
        use crate::image::sampling::test_fixtures::{fixture_pixels, fixture_texels};
        use crate::image::{ColourCache, ImageInterpolation, ImageStorage, Precision};
        use std::collections::BTreeMap;

        const TOLERANCE: f64 = 1.0e-6;
        let directory = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/repeat_edge");
        for (file, interpolation) in [
            (
                "blender_world_env_closest.json",
                ImageInterpolation::Closest,
            ),
            ("blender_world_env_linear.json", ImageInterpolation::Linear),
        ] {
            let document: serde_json::Value =
                serde_json::from_slice(&std::fs::read(directory.join(file))?)?;
            let width = usize::try_from(document["width"].as_u64().ok_or("width is missing")?)?;
            let height = usize::try_from(document["height"].as_u64().ok_or("height is missing")?)?;
            let blender = fixture_texels(&document["texels_blender_rows"])?;
            let pixels = fixture_pixels(width, height, &blender);
            let image = ImageData {
                width: u32::try_from(width)?,
                height: u32::try_from(height)?,
                pixels,
                tiles: BTreeMap::new(),
                interpolation,
                srgb: false,
                udim: false,
                storage: ImageStorage {
                    data: true,
                    straight_alpha: false,
                    ignore_alpha: false,
                    precision: Precision::Float,
                },
                colour: ColourCache::default(),
            };
            let mut worst = 0.0_f64;
            for record in document["records"]
                .as_array()
                .ok_or("records are missing")?
            {
                let value =
                    |index: usize| record[index].as_f64().ok_or("record value is not a number");
                let direction = DVec3::new(value(0)?, value(1)?, value(2)?);
                let expected = [value(3)?, value(4)?, value(5)?];
                let actual = environment_colour(&image, direction);
                for channel in 0..3 {
                    worst = worst.max((actual[channel] - expected[channel]).abs());
                }
            }
            assert!(
                worst <= TOLERANCE,
                "{file}: worst residual {worst:e} exceeds {TOLERANCE:e}"
            );
        }
        Ok(())
    }

    fn evaluate_mix_graph(mix: GraphNode) -> Result<DVec3> {
        let mix_output = if mix.node_type == "ShaderNodeMixRGB" {
            "Color"
        } else {
            "Result_Color"
        };
        let mix_id = Id::new("mix")?;
        let background_id = Id::new("background")?;
        let output_id = Id::new("output")?;
        let world_id = Id::new("world")?;
        let mut group = NodeGroup::new("world", GraphKind::Shader);
        group.nodes.insert(mix_id.clone(), mix);
        group.nodes.insert(
            background_id.clone(),
            GraphNode::new("ShaderNodeBackground"),
        );
        group
            .nodes
            .insert(output_id.clone(), GraphNode::new("ShaderNodeOutputWorld"));
        group.links.extend([
            GraphLink {
                from_node: mix_id,
                from_socket: mix_output.to_owned(),
                to_node: background_id.clone(),
                to_socket: "Color".to_owned(),
            },
            GraphLink {
                from_node: background_id,
                from_socket: "Background".to_owned(),
                to_node: output_id,
                to_socket: "Surface".to_owned(),
            },
        ]);
        let mut groups = Registry::new();
        groups.insert(world_id.clone(), group);
        let world = World {
            node_tree: Some(world_id),
            ..Default::default()
        };
        evaluate_world(
            &world,
            &groups,
            &std::collections::BTreeMap::new(),
            DVec3::Z,
            true,
        )
    }

    #[test]
    fn evaluates_rgba_generic_mix_sockets() -> Result<()> {
        let mut mix = GraphNode::new("ShaderNodeMix");
        mix.properties.insert("data_type".to_owned(), json!("RGBA"));
        mix.inputs.insert("Factor_Float".to_owned(), json!(0.25));
        mix.inputs
            .insert("A_Color".to_owned(), json!([1.0, 0.0, 0.0, 1.0]));
        mix.inputs
            .insert("B_Color".to_owned(), json!([0.0, 0.0, 1.0, 1.0]));

        assert_eq!(evaluate_mix_graph(mix)?, DVec3::new(0.75, 0.0, 0.25));
        Ok(())
    }

    #[test]
    fn rejects_default_float_generic_mix() -> Result<()> {
        let error = evaluate_mix_graph(GraphNode::new("ShaderNodeMix"))
            .err()
            .ok_or_else(|| PotError::new(ErrorCode::InvalidOperation, "mix graph was accepted"))?;

        assert_eq!(error.code, ErrorCode::UnsupportedFeature);
        assert!(error.message.contains("FLOAT"));
        Ok(())
    }

    #[test]
    fn keeps_legacy_mix_rgb_sockets() -> Result<()> {
        let mut mix = GraphNode::new("ShaderNodeMixRGB");
        mix.inputs.insert("Fac".to_owned(), json!(0.25));
        mix.inputs
            .insert("Color1".to_owned(), json!([1.0, 0.0, 0.0, 1.0]));
        mix.inputs
            .insert("Color2".to_owned(), json!([0.0, 0.0, 1.0, 1.0]));

        assert_eq!(evaluate_mix_graph(mix)?, DVec3::new(0.75, 0.0, 0.25));
        Ok(())
    }
}
