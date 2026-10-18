use std::collections::BTreeSet;

use glam::DVec3;
use serde_json::{Value as JsonValue, json};

use crate::{
    error::{ErrorCode, PotError, Result},
    graph::{GraphKind, GraphNode, NodeGroup},
    image::{ImageData, sample_with_interpolation},
    model::{Id, Registry, World},
};

#[derive(Clone, Copy)]
enum WorldValue {
    Scalar(f64),
    Color([f64; 4]),
    Vector(DVec3),
}

/// Evaluates the World node graph for a single outgoing direction.
pub fn evaluate_world(
    world: &World,
    groups: &Registry<NodeGroup>,
    images: &std::collections::BTreeMap<String, ImageData>,
    direction: DVec3,
) -> Result<DVec3> {
    let Some(graph_id) = &world.node_tree else {
        return Ok(DVec3::from_array(world.color) * world.strength);
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
        images,
        direction: normalize_or(direction, DVec3::Z),
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
    images: &'a std::collections::BTreeMap<String, ImageData>,
    direction: DVec3,
    active: BTreeSet<(Id, String)>,
}

impl WorldEvaluator<'_> {
    fn input(&mut self, id: &Id, socket: &str, fallback: WorldValue) -> Result<WorldValue> {
        if let Some(link) = self
            .group
            .links
            .iter()
            .find(|link| link.to_node == *id && socket_matches(&link.to_socket, socket))
        {
            return self.output(&link.from_node, &link.from_socket);
        }
        let node = self.node(id)?;
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
        let node = self.node(id)?.clone();
        let result = self.eval_node(id, &node, socket);
        self.active.remove(&key);
        result
    }

    fn node(&self, id: &Id) -> Result<&GraphNode> {
        self.group.nodes.get(id).ok_or_else(|| {
            PotError::new(
                ErrorCode::InvalidOperation,
                format!("world graph references missing node `{id}`"),
            )
        })
    }

    fn eval_node(&mut self, id: &Id, node: &GraphNode, socket: &str) -> Result<WorldValue> {
        match node.node_type.as_str() {
            "OutputWorld" | "ShaderNodeOutputWorld" => Err(PotError::new(
                ErrorCode::InvalidOperation,
                "world output node has no outputs",
            )),
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
                let direction = normalize_or(
                    as_vector(self.input(id, "Vector", WorldValue::Vector(self.direction))?),
                    self.direction,
                );
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
                let uv = [
                    direction.y.atan2(direction.x) / std::f64::consts::TAU + 0.5,
                    direction.z.clamp(-1.0, 1.0).acos() / std::f64::consts::PI,
                ];
                Ok(WorldValue::Color(sample_with_interpolation(
                    image,
                    // Potter's equirectangular v runs top-down, from the zenith; the sampler takes Blender's V-up coordinate.
                    [uv[0], 1.0 - uv[1]],
                    image.interpolation,
                )))
            }
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
                let data_type = node
                    .properties
                    .get("data_type")
                    .and_then(JsonValue::as_str)
                    .unwrap_or("FLOAT");
                if data_type != "RGBA" {
                    return Err(PotError::with_details(
                        ErrorCode::UnsupportedFeature,
                        format!("world ShaderNodeMix data type `{data_type}` is not supported"),
                        json!({
                            "feature_id": format!("world.node.ShaderNodeMix.data_type.{data_type}"),
                            "node_id": id,
                        }),
                    ));
                }
                let blend_type = node
                    .properties
                    .get("blend_type")
                    .and_then(JsonValue::as_str)
                    .unwrap_or("MIX");
                let factor = as_scalar(self.input(id, "Factor_Float", WorldValue::Scalar(0.5))?)
                    .clamp(0.0, 1.0);
                let first =
                    as_color(self.input(id, "A_Color", WorldValue::Color([0.5, 0.5, 0.5, 1.0]))?);
                let second =
                    as_color(self.input(id, "B_Color", WorldValue::Color([0.5, 0.5, 0.5, 1.0]))?);
                let blended = match blend_type {
                    "MIX" => second,
                    "ADD" => std::array::from_fn(|index| first[index] + second[index]),
                    "MULTIPLY" => std::array::from_fn(|index| first[index] * second[index]),
                    "SUBTRACT" => std::array::from_fn(|index| first[index] - second[index]),
                    "DIVIDE" => std::array::from_fn(|index| {
                        if second[index].abs() <= f64::EPSILON {
                            first[index]
                        } else {
                            first[index] / second[index]
                        }
                    }),
                    unsupported => {
                        return Err(PotError::with_details(
                            ErrorCode::UnsupportedFeature,
                            format!(
                                "world ShaderNodeMix blend type `{unsupported}` is not supported"
                            ),
                            json!({
                                "feature_id": format!("world.node.ShaderNodeMix.blend_type.{unsupported}"),
                                "node_id": id,
                            }),
                        ));
                    }
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{GraphKind, GraphLink};

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
