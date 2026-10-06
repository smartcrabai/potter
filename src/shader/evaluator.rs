use std::collections::{BTreeMap, BTreeSet};

use glam::{DQuat, DVec3, EulerRot};
use serde_json::{Value as JsonValue, json};

use crate::{
    error::{ErrorCode, PotError, Result},
    graph::{GraphKind, GraphNode, NodeGroup},
    image::{ImageData, ImageInterpolation, sample_with_interpolation},
    model::{Id, Material, Registry, TextureRef},
};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BsdfParams {
    pub base_color: [f64; 4],
    pub metallic: f64,
    pub roughness: f64,
    pub emission_color: [f64; 3],
    pub emission_strength: f64,
    pub transmission: f64,
    pub ior: f64,
    pub alpha: f64,
    pub subsurface_weight: f64,
    pub subsurface_radius: [f64; 3],
    pub subsurface_scale: f64,
    pub coat_weight: f64,
    pub coat_roughness: f64,
    pub coat_ior: f64,
    pub sheen_weight: f64,
    pub sheen_roughness: f64,
    pub sheen_tint: [f64; 3],
    pub specular_ior_level: f64,
    pub specular_tint: [f64; 3],
    pub anisotropic_ior: f64,
    pub anisotropic_rotation: f64,
    pub thin_film_thickness: f64,
    pub thin_film_ior: f64,
    pub volume_density: f64,
    pub volume_color: [f64; 3],
    pub volume_anisotropy: f64,
    pub volume_scattering: bool,
    pub volume_emission_color: [f64; 3],
    pub volume_emission_strength: f64,
    pub displacement_height: f64,
    pub normal: DVec3,
}

impl BsdfParams {
    fn from_material(material: &Material, normal: DVec3) -> Self {
        Self {
            base_color: material.base_color,
            metallic: material.metallic,
            roughness: material.roughness,
            emission_color: material.emission_color,
            emission_strength: material.emission_strength,
            transmission: material.transmission,
            ior: material.ior,
            alpha: material.base_color[3],
            subsurface_weight: 0.0,
            subsurface_radius: [1.0; 3],
            subsurface_scale: 0.05,
            coat_weight: 0.0,
            coat_roughness: 0.03,
            coat_ior: 1.5,
            sheen_weight: 0.0,
            sheen_roughness: 0.3,
            sheen_tint: [1.0; 3],
            specular_ior_level: 0.5,
            specular_tint: [1.0; 3],
            anisotropic_ior: 0.0,
            anisotropic_rotation: 0.0,
            thin_film_thickness: 0.0,
            thin_film_ior: 1.33,
            volume_density: material.volume_density.max(0.0),
            volume_color: material.volume_color,
            volume_anisotropy: material.volume_anisotropy.clamp(-0.999, 0.999),
            volume_scattering: false,
            volume_emission_color: material.emission_color,
            volume_emission_strength: material.emission_strength,
            displacement_height: 0.0,
            normal: normalize_or(normal, DVec3::Z),
        }
    }

    fn mix(first: Self, second: Self, factor: f64) -> Self {
        let factor = bounded_factor(factor);
        let mix_scalar = |a: f64, b: f64| a + (b - a) * factor;
        let mix3 = |a: [f64; 3], b: [f64; 3]| {
            [
                mix_scalar(a[0], b[0]),
                mix_scalar(a[1], b[1]),
                mix_scalar(a[2], b[2]),
            ]
        };
        let base_color = [
            mix_scalar(first.base_color[0], second.base_color[0]),
            mix_scalar(first.base_color[1], second.base_color[1]),
            mix_scalar(first.base_color[2], second.base_color[2]),
            mix_scalar(first.base_color[3], second.base_color[3]),
        ];
        Self {
            base_color,
            metallic: mix_scalar(first.metallic, second.metallic),
            roughness: mix_scalar(first.roughness, second.roughness),
            emission_color: mix3(first.emission_color, second.emission_color),
            emission_strength: mix_scalar(first.emission_strength, second.emission_strength),
            transmission: mix_scalar(first.transmission, second.transmission),
            ior: mix_scalar(first.ior, second.ior),
            alpha: mix_scalar(first.alpha, second.alpha),
            subsurface_weight: mix_scalar(first.subsurface_weight, second.subsurface_weight),
            subsurface_radius: mix3(first.subsurface_radius, second.subsurface_radius),
            subsurface_scale: mix_scalar(first.subsurface_scale, second.subsurface_scale),
            coat_weight: mix_scalar(first.coat_weight, second.coat_weight),
            coat_roughness: mix_scalar(first.coat_roughness, second.coat_roughness),
            coat_ior: mix_scalar(first.coat_ior, second.coat_ior),
            sheen_weight: mix_scalar(first.sheen_weight, second.sheen_weight),
            sheen_roughness: mix_scalar(first.sheen_roughness, second.sheen_roughness),
            sheen_tint: mix3(first.sheen_tint, second.sheen_tint),
            specular_ior_level: mix_scalar(first.specular_ior_level, second.specular_ior_level),
            specular_tint: mix3(first.specular_tint, second.specular_tint),
            anisotropic_ior: mix_scalar(first.anisotropic_ior, second.anisotropic_ior),
            anisotropic_rotation: mix_scalar(
                first.anisotropic_rotation,
                second.anisotropic_rotation,
            ),
            thin_film_thickness: mix_scalar(first.thin_film_thickness, second.thin_film_thickness),
            thin_film_ior: mix_scalar(first.thin_film_ior, second.thin_film_ior),
            volume_density: mix_scalar(first.volume_density, second.volume_density),
            volume_color: mix3(first.volume_color, second.volume_color),
            volume_anisotropy: mix_scalar(first.volume_anisotropy, second.volume_anisotropy),
            volume_scattering: first.volume_scattering || second.volume_scattering,
            volume_emission_color: mix3(first.volume_emission_color, second.volume_emission_color),
            volume_emission_strength: mix_scalar(
                first.volume_emission_strength,
                second.volume_emission_strength,
            ),
            displacement_height: mix_scalar(first.displacement_height, second.displacement_height),
            normal: normalize_or(first.normal.lerp(second.normal, factor), first.normal),
        }
    }

    fn add(first: Self, second: Self) -> Self {
        let add3 = |a: [f64; 3], b: [f64; 3]| [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
        Self {
            base_color: [
                first.base_color[0] + second.base_color[0],
                first.base_color[1] + second.base_color[1],
                first.base_color[2] + second.base_color[2],
                first.base_color[3].max(second.base_color[3]),
            ],
            metallic: first.metallic.max(second.metallic),
            roughness: first.roughness.min(second.roughness),
            emission_color: add3(first.emission_color, second.emission_color),
            emission_strength: first.emission_strength.max(second.emission_strength),
            transmission: first.transmission.max(second.transmission),
            ior: first.ior,
            alpha: first.alpha.max(second.alpha),
            subsurface_weight: first.subsurface_weight.max(second.subsurface_weight),
            subsurface_radius: add3(first.subsurface_radius, second.subsurface_radius),
            subsurface_scale: first.subsurface_scale.max(second.subsurface_scale),
            coat_weight: first.coat_weight.max(second.coat_weight),
            coat_roughness: first.coat_roughness.min(second.coat_roughness),
            coat_ior: first.coat_ior,
            sheen_weight: first.sheen_weight.max(second.sheen_weight),
            sheen_roughness: first.sheen_roughness.min(second.sheen_roughness),
            sheen_tint: add3(first.sheen_tint, second.sheen_tint),
            specular_ior_level: first.specular_ior_level.max(second.specular_ior_level),
            specular_tint: add3(first.specular_tint, second.specular_tint),
            anisotropic_ior: first.anisotropic_ior.max(second.anisotropic_ior),
            anisotropic_rotation: first.anisotropic_rotation,
            thin_film_thickness: first.thin_film_thickness.max(second.thin_film_thickness),
            thin_film_ior: first.thin_film_ior,
            volume_density: first.volume_density + second.volume_density,
            volume_color: add3(first.volume_color, second.volume_color),
            volume_anisotropy: first.volume_anisotropy,
            volume_scattering: first.volume_scattering || second.volume_scattering,
            volume_emission_color: add3(first.volume_emission_color, second.volume_emission_color),
            volume_emission_strength: first
                .volume_emission_strength
                .max(second.volume_emission_strength),
            displacement_height: first.displacement_height + second.displacement_height,
            normal: normalize_or(first.normal + second.normal, first.normal),
        }
    }
}

/// Per-hit coordinates and graph context used by procedural shader nodes.
#[derive(Clone, Copy, Debug)]
pub struct HitContext<'a> {
    pub node_groups: &'a Registry<NodeGroup>,
    /// Resolved linear image buffers keyed by their persistent image ID.
    pub image_data: Option<&'a BTreeMap<String, ImageData>>,
    pub uv: [f64; 2],
    pub uv_maps: Option<&'a [(&'a str, [f64; 2])]>,
    pub tangent_frame: Option<(DVec3, DVec3)>,
    pub generated: DVec3,
    pub position: DVec3,
    pub object: DVec3,
    pub normal: DVec3,
    pub view_direction: DVec3,
    pub color_attributes: Option<&'a BTreeMap<String, [f64; 4]>>,
}

impl<'a> HitContext<'a> {
    #[must_use]
    pub const fn new(node_groups: &'a Registry<NodeGroup>) -> Self {
        Self {
            image_data: None,
            node_groups,
            uv: [0.0; 2],
            uv_maps: None,
            tangent_frame: None,
            generated: DVec3::ZERO,
            position: DVec3::ZERO,
            object: DVec3::ZERO,
            normal: DVec3::Z,
            view_direction: DVec3::Z,
            color_attributes: None,
        }
    }

    /// Add pre-resolved pixels for Image Texture nodes and material texture references.
    #[must_use]
    pub const fn with_images(mut self, image_data: &'a BTreeMap<String, ImageData>) -> Self {
        self.image_data = Some(image_data);
        self
    }

    /// Provide per-hit UV coordinates by authored layer name without copying the map.
    #[must_use]
    pub const fn with_uv_maps(mut self, uv_maps: &'a [(&'a str, [f64; 2])]) -> Self {
        self.uv_maps = Some(uv_maps);
        self
    }

    /// Provide the per-hit tangent and bitangent for tangent-space normal mapping.
    #[must_use]
    pub const fn with_tangent_frame(mut self, tangent: DVec3, bitangent: DVec3) -> Self {
        self.tangent_frame = Some((tangent, bitangent));
        self
    }
}

/// Evaluate a material's shader graph at a surface hit. Materials without a node tree use their PBR fields.
///
/// # Errors
///
/// Returns `TARGET_NOT_FOUND` for a missing shader graph or image, `INVALID_OPERATION` for an invalid graph,
/// `EVALUATION_FAILED` for a cycle or missing material output, and `UNSUPPORTED_FEATURE` for unsupported
/// node behavior or unresolved image pixels and named UV maps in the hit context.
pub fn evaluate_surface(material: &Material, hit_ctx: &HitContext<'_>) -> Result<BsdfParams> {
    let Some(graph_id) = &material.node_tree else {
        let mut params = BsdfParams::from_material(material, hit_ctx.normal);
        apply_material_textures(material, hit_ctx, &mut params)?;
        return Ok(params);
    };
    let group = hit_ctx.node_groups.get(graph_id).ok_or_else(|| {
        PotError::with_details(
            ErrorCode::TargetNotFound,
            format!("shader graph `{graph_id}` was not found"),
            json!({"graph_id":graph_id}),
        )
    })?;
    if group.kind != GraphKind::Shader {
        return Err(PotError::new(
            ErrorCode::InvalidOperation,
            format!("material graph `{graph_id}` is not a shader graph"),
        ));
    }
    let mut fallback_output = None;
    let mut active_output = None;
    for (id, node) in &group.nodes {
        if matches!(
            node.node_type.as_str(),
            "OutputMaterial" | "ShaderNodeOutputMaterial"
        ) {
            if node
                .properties
                .get("is_active_output")
                .and_then(JsonValue::as_bool)
                == Some(true)
            {
                active_output = Some(id);
                break;
            }
            fallback_output.get_or_insert(id);
        }
    }
    let output = active_output.or(fallback_output).ok_or_else(|| {
        PotError::new(
            ErrorCode::EvaluationFailed,
            "shader graph has no OutputMaterial node",
        )
    })?;
    let has_volume = group
        .links
        .iter()
        .any(|link| link.to_node == *output && link.to_socket == "Volume");
    let has_displacement = group
        .links
        .iter()
        .any(|link| link.to_node == *output && link.to_socket == "Displacement");
    let mut evaluator = Evaluator {
        group,
        material,
        hit_ctx,
        active: BTreeSet::new(),
    };
    let Value::Shader(mut params) = evaluator.input(output, "Surface", ValueKind::Shader)? else {
        return Err(eval_error("material surface output is not a shader"));
    };
    if has_volume {
        let Value::Shader(volume) = evaluator.input(output, "Volume", ValueKind::Shader)? else {
            return Err(eval_error("material volume output is not a volume shader"));
        };
        params.volume_density = volume.volume_density;
        params.volume_color = volume.volume_color;
        params.volume_anisotropy = volume.volume_anisotropy;
        params.volume_scattering = volume.volume_scattering;
        params.volume_emission_color = volume.volume_emission_color;
        params.volume_emission_strength = volume.volume_emission_strength;
    }
    if has_displacement {
        params.displacement_height =
            match evaluator.input(output, "Displacement", ValueKind::Vector)? {
                Value::Vector(displacement) => displacement.dot(params.normal),
                Value::Scalar(displacement) => displacement,
                Value::Color(displacement) => {
                    DVec3::new(displacement[0], displacement[1], displacement[2]).dot(params.normal)
                }
                Value::Shader(_) => return Err(eval_error("material displacement is a shader")),
            };
    }
    apply_material_textures(material, hit_ctx, &mut params)?;
    Ok(*params)
}

/// Evaluates the connected `OutputMaterial` displacement input without evaluating surface textures.
pub fn evaluate_displacement(material: &Material, hit_ctx: &HitContext<'_>) -> Result<f64> {
    let Some(graph_id) = &material.node_tree else {
        return Ok(0.0);
    };
    let group = hit_ctx.node_groups.get(graph_id).ok_or_else(|| {
        PotError::with_details(
            ErrorCode::TargetNotFound,
            format!("shader graph `{graph_id}` was not found"),
            json!({"graph_id": graph_id}),
        )
    })?;
    if group.kind != GraphKind::Shader {
        return Err(PotError::new(
            ErrorCode::InvalidOperation,
            format!("material graph `{graph_id}` is not a shader graph"),
        ));
    }
    let mut fallback_output = None;
    let mut active_output = None;
    for (id, node) in &group.nodes {
        if matches!(
            node.node_type.as_str(),
            "OutputMaterial" | "ShaderNodeOutputMaterial"
        ) {
            if node
                .properties
                .get("is_active_output")
                .and_then(JsonValue::as_bool)
                == Some(true)
            {
                active_output = Some(id);
                break;
            }
            fallback_output.get_or_insert(id);
        }
    }
    let Some(output) = active_output.or(fallback_output) else {
        return Ok(0.0);
    };
    if !group
        .links
        .iter()
        .any(|link| link.to_node == *output && link.to_socket == "Displacement")
    {
        return Ok(0.0);
    }
    let mut evaluator = Evaluator {
        group,
        material,
        hit_ctx,
        active: BTreeSet::new(),
    };
    match evaluator.input(output, "Displacement", ValueKind::Vector)? {
        Value::Vector(displacement) => Ok(displacement.dot(hit_ctx.normal)),
        Value::Scalar(displacement) => Ok(displacement),
        Value::Color(displacement) => {
            Ok(DVec3::new(displacement[0], displacement[1], displacement[2]).dot(hit_ctx.normal))
        }
        Value::Shader(_) => Err(eval_error("material displacement is a shader")),
    }
}

fn apply_material_textures(
    material: &Material,
    hit_ctx: &HitContext<'_>,
    params: &mut BsdfParams,
) -> Result<()> {
    let has_textures = material.base_color_texture.is_some()
        || material.roughness_texture.is_some()
        || material.metallic_texture.is_some()
        || material.normal_texture.is_some();
    if !has_textures {
        return Ok(());
    }
    if let Some(texture) = &material.base_color_texture {
        let pixel = sample_material_texture(texture, hit_ctx)?;
        for (channel, texel) in params.base_color[..3].iter_mut().zip(pixel) {
            *channel *= texel;
        }
        params.alpha *= pixel[3];
    }
    if let Some(texture) = &material.roughness_texture {
        params.roughness *= sample_material_texture(texture, hit_ctx)?[0];
    }
    if let Some(texture) = &material.metallic_texture {
        params.metallic *= sample_material_texture(texture, hit_ctx)?[0];
    }
    if let Some(texture) = &material.normal_texture {
        let pixel = sample_material_texture(texture, hit_ctx)?;
        let tangent_normal = normalize_or(
            DVec3::new(
                pixel[0] * 2.0 - 1.0,
                pixel[1] * 2.0 - 1.0,
                pixel[2] * 2.0 - 1.0,
            ),
            DVec3::Z,
        );
        let normal = normalize_or(params.normal, hit_ctx.normal);
        let (tangent, bitangent) = tangent_basis_for_hit(hit_ctx, normal);
        params.normal = normalize_or(
            tangent * tangent_normal.x + bitangent * tangent_normal.y + normal * tangent_normal.z,
            normal,
        );
    }
    params.base_color = params.base_color.map(|value| value.clamp(0.0, 1.0));
    params.alpha = params.alpha.clamp(0.0, 1.0);
    params.metallic = params.metallic.clamp(0.0, 1.0);
    params.roughness = params.roughness.clamp(0.0, 1.0);
    Ok(())
}

fn sample_material_texture(texture: &TextureRef, hit_ctx: &HitContext<'_>) -> Result<[f64; 4]> {
    let image_id = texture.image.as_str();
    let images = hit_ctx.image_data.ok_or_else(|| {
        PotError::with_details(
            ErrorCode::UnsupportedFeature,
            "Material texture evaluation requires resolved image pixels in the hit context",
            json!({"feature_id":"shader.material_texture.image_data","image_id":image_id}),
        )
    })?;
    let image = images.get(image_id).ok_or_else(|| {
        PotError::with_details(
            ErrorCode::TargetNotFound,
            format!("image data `{image_id}` was not found in the hit context"),
            json!({"image_id":image_id}),
        )
    })?;
    let uv = if let Some(uv_map) = texture.uv_map.as_deref() {
        hit_ctx
            .uv_maps
            .and_then(|maps| {
                maps.iter()
                    .find(|(name, _)| *name == uv_map)
                    .map(|(_, coordinates)| *coordinates)
            })
            .ok_or_else(|| {
                PotError::with_details(
                    ErrorCode::UnsupportedFeature,
                    format!("UV map `{uv_map}` was not resolved in the hit context"),
                    json!({"feature_id":"shader.material_texture.uv_map","uv_map":uv_map,"image_id":image_id}),
                )
            })?
    } else {
        hit_ctx.uv
    };
    let tile = if image.tiles.is_empty() { 0 } else { 1001 };
    let pixel = sample_with_interpolation(image, uv, tile, texture.interpolation);
    if pixel.iter().any(|component| !component.is_finite()) {
        return Err(eval_error(format!(
            "Material texture `{image_id}` sampled non-finite pixels"
        )));
    }
    Ok(pixel)
}

#[derive(Clone, Debug)]
enum Value {
    Scalar(f64),
    Vector(DVec3),
    Color([f64; 4]),
    Shader(Box<BsdfParams>),
}

#[derive(Clone, Copy)]
enum ValueKind {
    Scalar,
    Vector,
    Color,
    Shader,
}

struct Evaluator<'graph, 'material, 'context, 'registry> {
    group: &'graph NodeGroup,
    material: &'material Material,
    hit_ctx: &'context HitContext<'registry>,
    active: BTreeSet<(Id, String)>,
}

impl Evaluator<'_, '_, '_, '_> {
    fn input(&mut self, node_id: &Id, socket: &str, kind: ValueKind) -> Result<Value> {
        let node = self.node(node_id)?;
        let incoming = self
            .group
            .links
            .iter()
            .find(|link| link.to_node == *node_id && socket_matches(&link.to_socket, socket));
        if let Some(link) = incoming {
            return self.output(&link.from_node, &link.from_socket);
        }
        let raw = node
            .inputs
            .iter()
            .find(|(name, _)| socket_matches(name, socket))
            .map(|(_, value)| value)
            .filter(|value| !value.is_null());
        let default = raw.map_or_else(|| socket_default(node, socket), Clone::clone);
        match kind {
            ValueKind::Scalar => Ok(Value::Scalar(number(&default, f64::NAN))),
            ValueKind::Vector => Ok(Value::Vector(vector(&default, DVec3::splat(f64::NAN)))),
            ValueKind::Color => Ok(Value::Color(color(&default, [f64::NAN; 4]))),
            ValueKind::Shader => match raw {
                Some(_) => Err(invalid("a shader socket cannot use a literal value")),
                None => Ok(Value::Shader(Box::new(BsdfParams::from_material(
                    self.material,
                    self.hit_ctx.normal,
                )))),
            },
        }
    }

    fn output(&mut self, node_id: &Id, socket: &str) -> Result<Value> {
        let key = (node_id.clone(), socket.to_owned());
        if !self.active.insert(key.clone()) {
            return Err(eval_error(format!(
                "shader graph cycle at node `{node_id}`"
            )));
        }
        let node = self.node(node_id)?.clone();
        let result = self.evaluate_node(node_id, &node, socket);
        self.active.remove(&key);
        result
    }

    fn node(&self, node_id: &Id) -> Result<&GraphNode> {
        self.group
            .nodes
            .get(node_id)
            .ok_or_else(|| invalid(format!("shader graph references absent node `{node_id}`")))
    }

    fn evaluate_node(&mut self, node_id: &Id, node: &GraphNode, socket: &str) -> Result<Value> {
        match node.node_type.as_str() {
            "OutputMaterial" | "ShaderNodeOutputMaterial" => {
                Err(invalid("material output has no output sockets"))
            }
            "ShaderNodeBsdfPrincipled" => Ok(Value::Shader(Box::new(self.principled(node_id)?))),
            "ShaderNodeVolumePrincipled" => {
                Ok(Value::Shader(Box::new(self.volume(node_id, true)?)))
            }
            "ShaderNodeVolumeAbsorption" => {
                Ok(Value::Shader(Box::new(self.volume(node_id, false)?)))
            }
            "ShaderNodeVolumeScatter" => Ok(Value::Shader(Box::new(self.volume(node_id, true)?))),
            "ShaderNodeDisplacement" => self.displacement(node_id, socket),
            "ShaderNodeTexImage" => self.image_texture(node_id, socket),
            "ShaderNodeEmission" => Ok(Value::Shader(Box::new(self.emission(node_id)?))),
            "ShaderNodeBsdfDiffuse" => Ok(Value::Shader(Box::new(self.diffuse(node_id)?))),
            "ShaderNodeBsdfGlossy" | "ShaderNodeBsdfAnisotropic" => {
                Ok(Value::Shader(Box::new(self.glossy(node_id)?)))
            }
            "ShaderNodeBsdfGlass" => Ok(Value::Shader(Box::new(self.glass(node_id)?))),
            "ShaderNodeBsdfTransparent" => Ok(Value::Shader(Box::new(self.transparent(node_id)?))),
            "ShaderNodeMixShader" => Ok(Value::Shader(Box::new(self.mix_shader(node_id)?))),
            "ShaderNodeAddShader" => Ok(Value::Shader(Box::new(self.add_shader(node_id)?))),
            "ShaderNodeBump" => self.bump(node_id, socket),
            "ShaderNodeTexChecker" => self.checker(node_id, socket),
            "ShaderNodeTexNoise" => self.noise(node_id, socket),
            "ShaderNodeTexVoronoi" => self.voronoi(node_id, socket),
            "ShaderNodeTexGradient" => self.gradient(node_id, socket),
            "ShaderNodeValToRGB" => self.color_ramp(node_id, socket),
            "ShaderNodeMixRGB" => self.mix_rgb(node_id, socket),
            "ShaderNodeMath" => self.math(node_id, socket),
            "ShaderNodeVectorMath" => self.vector_math(node_id, socket),
            "ShaderNodeMapping" => self.mapping(node_id, socket),
            "ShaderNodeTexCoord" => self.tex_coord(socket),
            "ShaderNodeNormalMap" => self.normal_map(node_id, socket),
            "ShaderNodeFresnel" => self.fresnel(node_id, socket),
            "ShaderNodeLayerWeight" => self.layer_weight(node_id, socket),
            "ShaderNodeAttribute" => self.attribute(node, socket),
            unknown => Err(PotError::with_details(
                ErrorCode::UnsupportedFeature,
                format!("shader node `{unknown}` is not supported by the native evaluator"),
                json!({"feature_id":format!("shader.node.{unknown}"),"node_id":node_id}),
            )),
        }
    }

    fn principled(&mut self, id: &Id) -> Result<BsdfParams> {
        let node = self.node(id)?;
        if node
            .properties
            .get("potter_simple_material")
            .and_then(JsonValue::as_bool)
            == Some(true)
        {
            return Ok(BsdfParams::from_material(
                self.material,
                self.hit_ctx.normal,
            ));
        }
        let mut params = BsdfParams::from_material(self.material, self.hit_ctx.normal);
        params.base_color = self.color_input(id, &["Base Color"], [0.8, 0.8, 0.8, 1.0])?;
        params.metallic = self.scalar_input(id, &["Metallic"], 0.0)?;
        params.roughness = self.scalar_input(id, &["Roughness"], 0.5)?;
        params.ior = self.scalar_input(id, &["IOR"], 1.5)?;
        params.alpha = self.scalar_input(id, &["Alpha"], 1.0)?;
        params.transmission =
            self.scalar_input(id, &["Transmission Weight", "Transmission"], 0.0)?;
        params.subsurface_weight =
            self.scalar_input(id, &["Subsurface Weight", "Subsurface"], 0.0)?;
        params.subsurface_radius = self
            .vector_input(id, &["Subsurface Radius"], DVec3::ONE)?
            .to_array();
        params.subsurface_scale = self.scalar_input(id, &["Subsurface Scale"], 0.05)?;
        params.coat_weight = self.scalar_input(id, &["Coat Weight", "Clearcoat"], 0.0)?;
        params.coat_roughness =
            self.scalar_input(id, &["Coat Roughness", "Clearcoat Roughness"], 0.03)?;
        params.coat_ior = self.scalar_input(id, &["Coat IOR"], 1.5)?;
        params.sheen_weight = self.scalar_input(id, &["Sheen Weight", "Sheen"], 0.0)?;
        params.sheen_roughness = self.scalar_input(id, &["Sheen Roughness"], 0.3)?;
        params.sheen_tint = self.color_input(id, &["Sheen Tint"], [1.0, 1.0, 1.0, 1.0])?[..3]
            .try_into()
            .map_err(|_| invalid("invalid sheen tint"))?;
        params.specular_ior_level =
            self.scalar_input(id, &["Specular IOR Level", "Specular"], 0.5)?;
        params.specular_tint = self.color_input(id, &["Specular Tint"], [1.0, 1.0, 1.0, 1.0])?[..3]
            .try_into()
            .map_err(|_| invalid("invalid specular tint"))?;
        params.anisotropic_ior =
            self.scalar_input(id, &["Anisotropic IOR Level", "Anisotropic"], 0.0)?;
        params.anisotropic_rotation = self.scalar_input(id, &["Anisotropic Rotation"], 0.0)?;
        params.thin_film_thickness = self.scalar_input(id, &["Thin Film Thickness"], 0.0)?;
        params.thin_film_ior = self.scalar_input(id, &["Thin Film IOR"], 1.33)?;
        params.emission_color =
            self.color_input(id, &["Emission Color", "Emission"], [0.0, 0.0, 0.0, 1.0])?[..3]
                .try_into()
                .map_err(|_| invalid("invalid emission color"))?;
        params.emission_strength = self.scalar_input(id, &["Emission Strength"], 0.0)?;
        params.normal = normalize_or(
            self.vector_input(id, &["Normal"], self.hit_ctx.normal)?,
            self.hit_ctx.normal,
        );
        Ok(params)
    }

    fn volume(&mut self, id: &Id, scattering: bool) -> Result<BsdfParams> {
        let mut params = BsdfParams::from_material(self.material, self.hit_ctx.normal);
        params.volume_density = self.scalar_input(id, &["Density"], 1.0)?.max(0.0);
        params.volume_color = self.color_input(id, &["Color"], [0.5, 0.5, 0.5, 1.0])?[..3]
            .try_into()
            .map_err(|_| invalid("invalid volume color"))?;
        params.volume_anisotropy = self
            .scalar_input(id, &["Anisotropy"], 0.0)?
            .clamp(-0.999, 0.999);
        params.volume_scattering = scattering;
        if scattering {
            params.volume_emission_color =
                self.color_input(id, &["Emission Color", "Emission"], [0.0, 0.0, 0.0, 1.0])?[..3]
                    .try_into()
                    .map_err(|_| invalid("invalid volume emission color"))?;
            params.volume_emission_strength = self.scalar_input(id, &["Emission Strength"], 0.0)?;
        } else {
            params.volume_emission_color = [0.0; 3];
            params.volume_emission_strength = 0.0;
        }
        Ok(params)
    }

    fn displacement(&mut self, id: &Id, socket: &str) -> Result<Value> {
        if !matches!(socket, "Displacement" | "Vector") {
            return Err(invalid(format!("unknown Displacement output `{socket}`")));
        }
        let height = self.scalar_input(id, &["Height"], 0.0)?;
        let midlevel = self.scalar_input(id, &["Midlevel"], self.material.displacement_midlevel)?;
        let scale = self.scalar_input(id, &["Scale"], self.material.displacement_scale)?;
        let normal = normalize_or(
            self.vector_input(id, &["Normal"], self.hit_ctx.normal)?,
            self.hit_ctx.normal,
        );
        Ok(Value::Vector(normal * ((height - midlevel) * scale)))
    }

    fn emission(&mut self, id: &Id) -> Result<BsdfParams> {
        let color = self.color_input(id, &["Color", "Emission"], [1.0, 1.0, 1.0, 1.0])?;
        let mut params = BsdfParams::from_material(self.material, self.hit_ctx.normal);
        params.base_color = color;
        params.emission_color = [color[0], color[1], color[2]];
        params.emission_strength = self.scalar_input(id, &["Strength"], 1.0)?;
        Ok(params)
    }

    fn diffuse(&mut self, id: &Id) -> Result<BsdfParams> {
        let mut params = BsdfParams::from_material(self.material, self.hit_ctx.normal);
        params.base_color = self.color_input(id, &["Color"], [0.8, 0.8, 0.8, 1.0])?;
        params.roughness = self.scalar_input(id, &["Roughness"], 0.0)?;
        params.normal = normalize_or(
            self.vector_input(id, &["Normal"], self.hit_ctx.normal)?,
            self.hit_ctx.normal,
        );
        Ok(params)
    }

    fn glossy(&mut self, id: &Id) -> Result<BsdfParams> {
        let mut params = self.diffuse(id)?;
        params.metallic = 1.0;
        params.roughness = self.scalar_input(id, &["Roughness"], 0.5)?;
        Ok(params)
    }

    fn glass(&mut self, id: &Id) -> Result<BsdfParams> {
        let mut params = self.diffuse(id)?;
        params.transmission = 1.0;
        params.ior = self.scalar_input(id, &["IOR"], 1.45)?;
        params.roughness = self.scalar_input(id, &["Roughness"], 0.0)?;
        Ok(params)
    }

    fn transparent(&mut self, id: &Id) -> Result<BsdfParams> {
        let mut params = self.diffuse(id)?;
        params.base_color[3] = 0.0;
        params.alpha = 0.0;
        params.transmission = 1.0;
        Ok(params)
    }

    fn mix_shader(&mut self, id: &Id) -> Result<BsdfParams> {
        let factor = self.scalar_input(id, &["Fac"], 0.5)?;
        let first = self.shader_input(id, &["Shader", "Shader 1"])?;
        let second = self.shader_input(id, &["Shader_001", "Shader 2"])?;
        Ok(BsdfParams::mix(first, second, factor))
    }

    fn add_shader(&mut self, id: &Id) -> Result<BsdfParams> {
        let first = self.shader_input(id, &["Shader", "Shader 1"])?;
        let second = self.shader_input(id, &["Shader_001", "Shader 2"])?;
        Ok(BsdfParams::add(first, second))
    }

    fn shader_input(&mut self, id: &Id, names: &[&str]) -> Result<BsdfParams> {
        match self.input(
            id,
            names.first().copied().unwrap_or("Shader"),
            ValueKind::Shader,
        )? {
            Value::Shader(value) => Ok(*value),
            _ => Err(invalid("shader socket type mismatch")),
        }
    }

    fn image_texture(&mut self, id: &Id, socket: &str) -> Result<Value> {
        let uv = self.vector_input(
            id,
            &["Vector"],
            DVec3::new(self.hit_ctx.uv[0], self.hit_ctx.uv[1], 0.0),
        )?;
        let node = self.node(id)?;
        let image_id = node
            .properties
            .get("image")
            .or_else(|| node.properties.get("image_id"))
            .and_then(JsonValue::as_str)
            .ok_or_else(|| invalid(format!("Image Texture node `{id}` requires an image ID")))?;
        if !Id::is_valid(image_id) {
            return Err(invalid(format!(
                "Image Texture node `{id}` has an invalid image ID"
            )));
        }
        let images = self.hit_ctx.image_data.ok_or_else(|| {
            PotError::with_details(
                ErrorCode::UnsupportedFeature,
                "Image Texture evaluation requires resolved image pixels in the hit context",
                json!({"feature_id":"shader.node.ShaderNodeTexImage.image_data","node_id":id,"image_id":image_id}),
            )
        })?;
        let image = images.get(image_id).ok_or_else(|| {
            PotError::with_details(
                ErrorCode::TargetNotFound,
                format!("image data `{image_id}` was not found in the hit context"),
                json!({"image_id":image_id,"node_id":id}),
            )
        })?;
        let tile = ["tile", "image_user_tile", "udim_tile"]
            .iter()
            .find_map(|key| node.properties.get(*key).and_then(JsonValue::as_u64))
            .and_then(|value| u32::try_from(value).ok())
            .unwrap_or(if image.tiles.is_empty() { 0 } else { 1001 });
        let interpolation = match node
            .properties
            .get("interpolation")
            .and_then(JsonValue::as_str)
        {
            Some("linear" | "Linear" | "LINEAR") => ImageInterpolation::Linear,
            Some("closest" | "Closest" | "CLOSEST") => ImageInterpolation::Closest,
            Some(interpolation) => {
                return Err(PotError::with_details(
                    ErrorCode::UnsupportedFeature,
                    format!("Image Texture interpolation `{interpolation}` is not supported"),
                    json!({"feature_id":"shader.node.ShaderNodeTexImage.interpolation","node_id":id,"interpolation":interpolation}),
                ));
            }
            None => image.interpolation,
        };
        let pixel = sample_with_interpolation(image, [uv.x, uv.y], tile, interpolation);
        if pixel.iter().any(|component| !component.is_finite()) {
            return Err(eval_error(format!(
                "Image Texture node `{id}` sampled non-finite pixels"
            )));
        }
        match socket {
            "Color" => Ok(Value::Color(pixel)),
            "Alpha" => Ok(Value::Scalar(pixel[3])),
            _ => Err(invalid(format!("unknown Image Texture output `{socket}`"))),
        }
    }

    fn bump(&mut self, id: &Id, socket: &str) -> Result<Value> {
        if socket != "Normal" {
            return Err(invalid(format!("unknown Bump output `{socket}`")));
        }
        let normal = normalize_or(
            self.vector_input(id, &["Normal"], self.hit_ctx.normal)?,
            self.hit_ctx.normal,
        );
        let strength = self.scalar_input(id, &["Strength"], 1.0)?;
        let distance = self.scalar_input(id, &["Distance"], 1.0)?;
        if strength.abs() <= f64::EPSILON || distance.abs() <= f64::EPSILON {
            return Ok(Value::Vector(normal));
        }
        let (tangent_u, tangent_v) = tangent_basis_for_hit(self.hit_ctx, normal);
        let epsilon = 1.0e-3;
        let mut plus_u = *self.hit_ctx;
        let mut minus_u = *self.hit_ctx;
        let mut plus_v = *self.hit_ctx;
        let mut minus_v = *self.hit_ctx;
        offset_context(&mut plus_u, 0, tangent_u, epsilon);
        offset_context(&mut minus_u, 0, tangent_u, -epsilon);
        offset_context(&mut plus_v, 1, tangent_v, epsilon);
        offset_context(&mut minus_v, 1, tangent_v, -epsilon);
        let derivative_u =
            (self.height_at(id, &plus_u)? - self.height_at(id, &minus_u)?) / (2.0 * epsilon);
        let derivative_v =
            (self.height_at(id, &plus_v)? - self.height_at(id, &minus_v)?) / (2.0 * epsilon);
        let invert = self
            .node(id)?
            .properties
            .get("invert")
            .and_then(JsonValue::as_bool)
            .unwrap_or(false);
        let sign = if invert { -1.0 } else { 1.0 };
        let perturbed = normal
            - (tangent_u * derivative_u + tangent_v * derivative_v) * (strength * distance * sign);
        Ok(Value::Vector(normalize_or(perturbed, normal)))
    }

    fn height_at(&self, id: &Id, context: &HitContext<'_>) -> Result<f64> {
        let mut evaluator = Evaluator {
            group: self.group,
            material: self.material,
            hit_ctx: context,
            active: self.active.clone(),
        };
        evaluator.scalar_input(id, &["Height"], 0.5)
    }
    fn checker(&mut self, id: &Id, socket: &str) -> Result<Value> {
        let vector = self.vector_input(
            id,
            &["Vector"],
            DVec3::new(self.hit_ctx.uv[0], self.hit_ctx.uv[1], 0.0),
        )?;
        let scale = self.scalar_input(id, &["Scale"], 5.0)?;
        let scaled = vector * scale;
        let parity = scaled.x.floor() + scaled.y.floor() + scaled.z.floor();
        let first = self.color_input(id, &["Color1"], [0.2, 0.2, 0.2, 1.0])?;
        let second = self.color_input(id, &["Color2"], [0.8, 0.8, 0.8, 1.0])?;
        let first_chosen = parity.rem_euclid(2.0) < 1.0;
        let chosen = if first_chosen { first } else { second };
        match socket {
            "Color" => Ok(Value::Color(chosen)),
            "Fac" | "Factor" => Ok(Value::Scalar(if first_chosen { 0.0 } else { 1.0 })),
            _ => Err(invalid(format!(
                "unknown Checker Texture output `{socket}`"
            ))),
        }
    }

    fn noise(&mut self, id: &Id, socket: &str) -> Result<Value> {
        let vector = self.vector_input(id, &["Vector"], self.hit_ctx.generated)?;
        let scale = self.scalar_input(id, &["Scale"], 5.0)?;
        let detail = self.scalar_input(id, &["Detail"], 2.0)?.clamp(0.0, 8.0);
        let roughness = self.scalar_input(id, &["Roughness"], 0.5)?.clamp(0.0, 1.0);
        let mut value = 0.0;
        let mut amplitude = 1.0;
        let mut total_amplitude = 0.0;
        let mut frequency = scale;
        for _ in 0..=detail.floor() as usize {
            value += perlin(vector * frequency) * amplitude;
            total_amplitude += amplitude;
            frequency *= 2.0;
            amplitude *= roughness;
        }
        let factor = (0.5 + 0.5 * value / total_amplitude.max(f64::MIN_POSITIVE)).clamp(0.0, 1.0);
        match socket {
            "Fac" | "Factor" => Ok(Value::Scalar(factor)),
            "Color" => Ok(Value::Color([factor, factor, factor, 1.0])),
            _ => Err(invalid(format!("unknown Noise Texture output `{socket}`"))),
        }
    }

    fn voronoi(&mut self, id: &Id, socket: &str) -> Result<Value> {
        let vector = self.vector_input(id, &["Vector"], self.hit_ctx.generated)?
            * self.scalar_input(id, &["Scale"], 5.0)?;
        let cell = vector.floor();
        let mut nearest = f64::INFINITY;
        let mut nearest_point = DVec3::ZERO;
        for z in -1..=1 {
            for y in -1..=1 {
                for x in -1..=1 {
                    let coordinate = cell + DVec3::new(f64::from(x), f64::from(y), f64::from(z));
                    let offset = DVec3::new(
                        lattice_random(coordinate, 0),
                        lattice_random(coordinate, 1),
                        lattice_random(coordinate, 2),
                    );
                    let point = coordinate + offset;
                    let distance = (vector - point).length();
                    if distance < nearest {
                        nearest = distance;
                        nearest_point = point;
                    }
                }
            }
        }
        match socket {
            "Distance" | "Fac" => Ok(Value::Scalar(nearest)),
            "Color" => Ok(Value::Color([
                lattice_random(nearest_point, 3),
                lattice_random(nearest_point, 4),
                lattice_random(nearest_point, 5),
                1.0,
            ])),
            "Position" => Ok(Value::Vector(nearest_point)),
            _ => Err(invalid(format!(
                "unknown Voronoi Texture output `{socket}`"
            ))),
        }
    }

    fn gradient(&mut self, id: &Id, socket: &str) -> Result<Value> {
        let vector = self.vector_input(id, &["Vector"], self.hit_ctx.generated)?;
        let gradient_type = self
            .node(id)?
            .properties
            .get("gradient_type")
            .and_then(JsonValue::as_str)
            .unwrap_or("LINEAR");
        let factor = match gradient_type {
            "LINEAR" => vector.x,
            "QUADRATIC" => vector.x.max(0.0).powi(2),
            "EASING" => {
                let x = vector.x.clamp(0.0, 1.0);
                x * x * (3.0 - 2.0 * x)
            }
            "SPHERICAL" => vector.length(),
            "QUADRATIC_SPHERE" => vector.length_squared(),
            "RADIAL" => vector.y.atan2(vector.x) / std::f64::consts::TAU + 0.5,
            other => return Err(unsupported_node(id, &format!("Gradient:{other}"))),
        }
        .clamp(0.0, 1.0);
        match socket {
            "Fac" | "Factor" => Ok(Value::Scalar(factor)),
            "Color" => Ok(Value::Color([factor, factor, factor, 1.0])),
            _ => Err(invalid(format!(
                "unknown Gradient Texture output `{socket}`"
            ))),
        }
    }

    fn color_ramp(&mut self, id: &Id, socket: &str) -> Result<Value> {
        let factor = self.scalar_input(id, &["Fac"], 0.0)?.clamp(0.0, 1.0);
        let node = self.node(id)?;
        let result = if let Some(elements) = node
            .properties
            .get("elements")
            .and_then(JsonValue::as_array)
        {
            if elements.len() < 2 {
                return Err(invalid("ColorRamp requires at least two color elements"));
            }
            let interpolation = node
                .properties
                .get("interpolation")
                .and_then(JsonValue::as_str)
                .unwrap_or("LINEAR");
            let mut previous = ramp_element(&elements[0])?;
            if factor <= previous.0 {
                previous.1
            } else {
                let mut result = previous.1;
                for raw in elements.iter().skip(1) {
                    let next = ramp_element(raw)?;
                    if next.0 < previous.0 {
                        return Err(invalid("ColorRamp element positions must be ordered"));
                    }
                    if factor <= next.0 {
                        let local_factor = if next.0 - previous.0 <= f64::EPSILON {
                            1.0
                        } else {
                            ((factor - previous.0) / (next.0 - previous.0)).clamp(0.0, 1.0)
                        };
                        let local_factor = match interpolation {
                            "LINEAR" => local_factor,
                            "EASE" => local_factor * local_factor * (3.0 - 2.0 * local_factor),
                            "CONSTANT" => 0.0,
                            other => {
                                return Err(unsupported_node(id, &format!("ColorRamp:{other}")));
                            }
                        };
                        result = color_lerp(previous.1, next.1, local_factor);
                        break;
                    }
                    previous = next;
                    result = previous.1;
                }
                result
            }
        } else {
            let first = node
                .properties
                .get("color1")
                .map_or([0.0, 0.0, 0.0, 1.0], |value| {
                    color(value, [0.0, 0.0, 0.0, 1.0])
                });
            let second = node
                .properties
                .get("color2")
                .map_or([1.0; 4], |value| color(value, [1.0; 4]));
            color_lerp(first, second, factor)
        };
        match socket {
            "Color" => Ok(Value::Color(result)),
            "Alpha" => Ok(Value::Scalar(result[3])),
            _ => Err(invalid(format!("unknown ColorRamp output `{socket}`"))),
        }
    }

    fn mix_rgb(&mut self, id: &Id, socket: &str) -> Result<Value> {
        let factor = self.scalar_input(id, &["Fac"], 0.5)?.clamp(0.0, 1.0);
        let first = self.color_input(id, &["Color1"], [0.5, 0.5, 0.5, 1.0])?;
        let second = self.color_input(id, &["Color2"], [0.5, 0.5, 0.5, 1.0])?;
        let blend = self
            .node(id)?
            .properties
            .get("blend_type")
            .and_then(JsonValue::as_str)
            .unwrap_or("MIX");
        let blended = match blend {
            "MIX" => second,
            "ADD" => color_binary(first, second, |a, b| a + b),
            "MULTIPLY" => color_binary(first, second, |a, b| a * b),
            "SUBTRACT" => color_binary(first, second, |a, b| a - b),
            "DIVIDE" => color_binary(first, second, |a, b| {
                if b.abs() <= f64::EPSILON { a } else { a / b }
            }),
            other => return Err(unsupported_node(id, &format!("MixRGB:{other}"))),
        };
        let result = color_lerp(first, blended, factor);
        match socket {
            "Color" => Ok(Value::Color(result)),
            "Alpha" => Ok(Value::Scalar(result[3])),
            _ => Err(invalid(format!("unknown MixRGB output `{socket}`"))),
        }
    }

    fn math(&mut self, id: &Id, socket: &str) -> Result<Value> {
        let first = self.scalar_input(id, &["Value", "Value_001"], 0.0)?;
        let second = self.scalar_input(id, &["Value_001", "Value_002"], 0.0)?;
        let third = self.scalar_input(id, &["Value_002", "Value_003"], 0.0)?;
        let operation = self
            .node(id)?
            .properties
            .get("operation")
            .and_then(JsonValue::as_str)
            .unwrap_or("ADD");
        let result = match operation {
            "ADD" => first + second,
            "SUBTRACT" => first - second,
            "MULTIPLY" => first * second,
            "DIVIDE" => {
                if second.abs() <= f64::EPSILON {
                    0.0
                } else {
                    first / second
                }
            }
            "MULTIPLY_ADD" => first * second + third,
            "POWER" => first.powf(second),
            "SINE" => first.sin(),
            "COSINE" => first.cos(),
            "TANGENT" => first.tan(),
            "SQRT" => first.max(0.0).sqrt(),
            "ABSOLUTE" => first.abs(),
            "MINIMUM" => first.min(second),
            "MAXIMUM" => first.max(second),
            "LESS_THAN" => f64::from(first < second),
            "GREATER_THAN" => f64::from(first > second),
            "COMPARE" => f64::from((first - second).abs() <= 1.0e-6),
            "PINGPONG" => {
                let scale = second.abs().max(f64::MIN_POSITIVE);
                let value = first.rem_euclid(2.0 * scale);
                scale - (value - scale).abs()
            }
            other => return Err(unsupported_node(id, &format!("Math:{other}"))),
        };
        if !result.is_finite() {
            return Err(eval_error(format!(
                "Math node `{id}` produced a non-finite result"
            )));
        }
        match socket {
            "Value" => Ok(Value::Scalar(result)),
            _ => Err(invalid(format!("unknown Math output `{socket}`"))),
        }
    }

    fn vector_math(&mut self, id: &Id, socket: &str) -> Result<Value> {
        let first = self.vector_input(id, &["Vector", "Vector_001"], DVec3::ZERO)?;
        let second = self.vector_input(id, &["Vector_001", "Vector_002"], DVec3::ZERO)?;
        let operation = self
            .node(id)?
            .properties
            .get("operation")
            .and_then(JsonValue::as_str)
            .unwrap_or("ADD");
        let (vector, scalar) = match operation {
            "ADD" => (first + second, None),
            "SUBTRACT" => (first - second, None),
            "MULTIPLY" => (first * second, None),
            "CROSS_PRODUCT" => (first.cross(second), None),
            "DOT_PRODUCT" => (DVec3::ZERO, Some(first.dot(second))),
            "LENGTH" => (DVec3::ZERO, Some(first.length())),
            "DISTANCE" => (DVec3::ZERO, Some(first.distance(second))),
            "NORMALIZE" => (normalize_or(first, DVec3::ZERO), None),
            "SCALE" => (first * self.scalar_input(id, &["Scale"], 1.0)?, None),
            other => return Err(unsupported_node(id, &format!("VectorMath:{other}"))),
        };
        if !vector.is_finite() || scalar.is_some_and(|value| !value.is_finite()) {
            return Err(eval_error(format!(
                "Vector Math node `{id}` produced a non-finite result"
            )));
        }
        if socket == "Value" {
            scalar
                .map(Value::Scalar)
                .ok_or_else(|| invalid("Vector Math node has no scalar result"))
        } else if socket == "Vector" {
            Ok(Value::Vector(vector))
        } else {
            Err(invalid(format!("unknown Vector Math output `{socket}`")))
        }
    }

    fn mapping(&mut self, id: &Id, socket: &str) -> Result<Value> {
        let vector = self.vector_input(id, &["Vector"], self.hit_ctx.generated)?;
        let location = self.vector_input(id, &["Location"], DVec3::ZERO)?;
        let rotation = self.vector_input(id, &["Rotation"], DVec3::ZERO)?;
        let scale = self.vector_input(id, &["Scale"], DVec3::ONE)?;
        let mapped = location
            + DQuat::from_euler(EulerRot::XYZ, rotation.x, rotation.y, rotation.z)
                * (vector * scale);
        match socket {
            "Vector" => Ok(Value::Vector(mapped)),
            _ => Err(invalid(format!("unknown Mapping output `{socket}`"))),
        }
    }

    fn tex_coord(&self, socket: &str) -> Result<Value> {
        let value = match socket {
            "Generated" => self.hit_ctx.generated,
            "UV" => DVec3::new(self.hit_ctx.uv[0], self.hit_ctx.uv[1], 0.0),
            "Object" => self.hit_ctx.object,
            "Normal" => self.hit_ctx.normal,
            "Position" => self.hit_ctx.position,
            "Camera" | "Window" | "Reflection" => {
                return Err(unsupported_node_id("TexCoord", socket));
            }
            _ => {
                return Err(invalid(format!(
                    "unknown Texture Coordinate output `{socket}`"
                )));
            }
        };
        Ok(Value::Vector(value))
    }

    fn normal_map(&mut self, id: &Id, socket: &str) -> Result<Value> {
        if socket != "Normal" {
            return Err(invalid(format!("unknown Normal Map output `{socket}`")));
        }
        let space = self
            .node(id)?
            .properties
            .get("space")
            .and_then(JsonValue::as_str)
            .unwrap_or("TANGENT_SPACE");
        if !matches!(space, "TANGENT_SPACE" | "TANGENT") {
            return Err(unsupported_node(id, &format!("NormalMap:{space}")));
        }
        let color = self.color_input(id, &["Color"], [0.5, 0.5, 1.0, 1.0])?;
        let tangent_normal = DVec3::new(
            color[0] * 2.0 - 1.0,
            color[1] * 2.0 - 1.0,
            color[2] * 2.0 - 1.0,
        );
        let base_normal = normalize_or(self.hit_ctx.normal, DVec3::Z);
        let (tangent_u, tangent_v) = tangent_basis_for_hit(self.hit_ctx, base_normal);
        let mapped_normal = normalize_or(
            tangent_u * tangent_normal.x
                + tangent_v * tangent_normal.y
                + base_normal * tangent_normal.z,
            base_normal,
        );
        let strength = self.scalar_input(id, &["Strength"], 1.0)?.clamp(0.0, 1.0);
        Ok(Value::Vector(normalize_or(
            base_normal.lerp(mapped_normal, strength),
            base_normal,
        )))
    }

    fn fresnel(&mut self, id: &Id, socket: &str) -> Result<Value> {
        let ior = self.scalar_input(id, &["IOR"], 1.45)?.max(1.0);
        let cos_theta = normalize_or(self.hit_ctx.normal, DVec3::Z)
            .dot(normalize_or(self.hit_ctx.view_direction, DVec3::Z))
            .abs()
            .clamp(0.0, 1.0);
        let r0 = ((ior - 1.0) / (ior + 1.0)).powi(2);
        let facing = r0 + (1.0 - r0) * (1.0 - cos_theta).powi(5);
        match socket {
            "Fac" | "Factor" => Ok(Value::Scalar(facing)),
            _ => Err(invalid(format!("unknown Fresnel output `{socket}`"))),
        }
    }

    fn layer_weight(&mut self, id: &Id, socket: &str) -> Result<Value> {
        let blend = self.scalar_input(id, &["Blend"], 0.5)?.clamp(0.0, 1.0);
        let cos_theta = normalize_or(self.hit_ctx.normal, DVec3::Z)
            .dot(normalize_or(self.hit_ctx.view_direction, DVec3::Z))
            .abs()
            .clamp(0.0, 1.0);
        let facing = cos_theta.powf(blend.max(0.001));
        match socket {
            "Facing" => Ok(Value::Scalar(facing)),
            "Fresnel" => Ok(Value::Scalar(1.0 - facing)),
            _ => Err(invalid(format!("unknown Layer Weight output `{socket}`"))),
        }
    }

    fn attribute(&self, node: &GraphNode, socket: &str) -> Result<Value> {
        let name = node
            .properties
            .get("attribute_name")
            .or_else(|| node.properties.get("name"))
            .and_then(JsonValue::as_str)
            .ok_or_else(|| invalid("Attribute node requires attribute_name"))?;
        let color = self
            .hit_ctx
            .color_attributes
            .and_then(|attributes| attributes.get(name))
            .copied()
            .unwrap_or([0.0, 0.0, 0.0, 1.0]);
        match socket {
            "Color" => Ok(Value::Color(color)),
            "Alpha" => Ok(Value::Scalar(color[3])),
            _ => Err(invalid(format!("unknown Attribute output `{socket}`"))),
        }
    }

    fn scalar_input(&mut self, id: &Id, names: &[&str], default: f64) -> Result<f64> {
        match self.input(
            id,
            names.first().copied().unwrap_or("Value"),
            ValueKind::Scalar,
        )? {
            Value::Scalar(value) => Ok(value),
            Value::Vector(value) => Ok(value.x),
            Value::Color(value) => Ok(value[0]),
            Value::Shader(_) => Err(invalid("expected a scalar shader socket")),
        }
        .map(|value| if value.is_finite() { value } else { default })
    }

    fn vector_input(&mut self, id: &Id, names: &[&str], default: DVec3) -> Result<DVec3> {
        match self.input(
            id,
            names.first().copied().unwrap_or("Vector"),
            ValueKind::Vector,
        )? {
            Value::Vector(value) => Ok(value),
            Value::Scalar(value) => Ok(DVec3::splat(value)),
            Value::Color(value) => Ok(DVec3::new(value[0], value[1], value[2])),
            Value::Shader(_) => Err(invalid("expected a vector shader socket")),
        }
        .map(|value| if value.is_finite() { value } else { default })
    }

    fn color_input(&mut self, id: &Id, names: &[&str], default: [f64; 4]) -> Result<[f64; 4]> {
        match self.input(
            id,
            names.first().copied().unwrap_or("Color"),
            ValueKind::Color,
        )? {
            Value::Color(value) => Ok(value),
            Value::Scalar(value) => Ok([value, value, value, 1.0]),
            Value::Vector(value) => Ok([value.x, value.y, value.z, 1.0]),
            Value::Shader(_) => Err(invalid("expected a color shader socket")),
        }
        .map(|value| {
            if value.iter().all(|part| part.is_finite()) {
                value
            } else {
                default
            }
        })
    }
}

fn offset_context(context: &mut HitContext<'_>, axis: usize, direction: DVec3, distance: f64) {
    context.uv[axis] += distance;
    context.generated += direction * distance;
    context.position += direction * distance;
    context.object += direction * distance;
}

fn socket_default(node: &GraphNode, socket: &str) -> JsonValue {
    let alias = match socket {
        "Base Color" => "Base Color",
        "Transmission Weight" => "Transmission",
        "Sheen Weight" => "Sheen",
        "Coat Weight" => "Clearcoat",
        "Emission Color" => "Emission",
        "Shader" => "Shader",
        _ => socket,
    };
    node.inputs.get(alias).cloned().unwrap_or(JsonValue::Null)
}

fn socket_matches(candidate: &str, requested: &str) -> bool {
    if candidate == requested {
        return true;
    }
    matches!(
        (candidate, requested),
        ("Shader", "Shader 1")
            | ("Shader 1", "Shader")
            | ("Shader_001", "Shader 2")
            | ("Shader 2", "Shader_001")
            | ("Transmission", "Transmission Weight")
            | ("Transmission Weight", "Transmission")
            | ("Subsurface", "Subsurface Weight")
            | ("Subsurface Weight", "Subsurface")
            | ("Clearcoat", "Coat Weight")
            | ("Coat Weight", "Clearcoat")
            | ("Sheen", "Sheen Weight")
            | ("Sheen Weight", "Sheen")
            | ("Emission", "Emission Color")
            | ("Emission Color", "Emission")
    )
}

fn number(value: &JsonValue, default: f64) -> f64 {
    value
        .as_f64()
        .filter(|value| value.is_finite())
        .unwrap_or(default)
}

fn vector(value: &JsonValue, default: DVec3) -> DVec3 {
    let Some(values) = value.as_array() else {
        return default;
    };
    if values.len() < 3 {
        return default;
    }
    let Some(x) = values[0].as_f64() else {
        return default;
    };
    let Some(y) = values[1].as_f64() else {
        return default;
    };
    let Some(z) = values[2].as_f64() else {
        return default;
    };
    let result = DVec3::new(x, y, z);
    if result.is_finite() { result } else { default }
}

fn color(value: &JsonValue, default: [f64; 4]) -> [f64; 4] {
    let Some(values) = value.as_array() else {
        return default;
    };
    if values.len() < 3 {
        return default;
    }
    let Some(red) = values[0].as_f64() else {
        return default;
    };
    let Some(green) = values[1].as_f64() else {
        return default;
    };
    let Some(blue) = values[2].as_f64() else {
        return default;
    };
    let alpha = values.get(3).and_then(JsonValue::as_f64).unwrap_or(1.0);
    let result = [red, green, blue, alpha];
    if result.iter().all(|part| part.is_finite()) {
        result
    } else {
        default
    }
}

fn ramp_element(value: &JsonValue) -> Result<(f64, [f64; 4])> {
    let object = value
        .as_object()
        .ok_or_else(|| invalid("ColorRamp elements must be objects"))?;
    let position = object
        .get("position")
        .or_else(|| object.get("pos"))
        .and_then(JsonValue::as_f64)
        .filter(|position| position.is_finite())
        .ok_or_else(|| invalid("ColorRamp element position must be finite"))?;
    let color_value = object
        .get("color")
        .ok_or_else(|| invalid("ColorRamp element requires a color"))?;
    let color = color(color_value, [f64::NAN; 4]);
    if color.iter().any(|component| !component.is_finite()) {
        return Err(invalid("ColorRamp element color must be finite RGBA"));
    }
    Ok((position, color))
}

fn color_lerp(first: [f64; 4], second: [f64; 4], factor: f64) -> [f64; 4] {
    let factor = bounded_factor(factor);
    std::array::from_fn(|index| first[index] + (second[index] - first[index]) * factor)
}

fn color_binary(first: [f64; 4], second: [f64; 4], f: impl Fn(f64, f64) -> f64) -> [f64; 4] {
    std::array::from_fn(|index| f(first[index], second[index]))
}

fn bounded_factor(value: f64) -> f64 {
    value.clamp(0.0, 1.0)
}

fn normalize_or(value: DVec3, fallback: DVec3) -> DVec3 {
    if value.is_finite() && value.length_squared() > f64::EPSILON {
        value.normalize()
    } else {
        fallback
    }
}

fn tangent_basis(normal: DVec3) -> (DVec3, DVec3) {
    let tangent_u = if normal.z.abs() > 0.999 {
        DVec3::X
    } else {
        normalize_or(DVec3::Z.cross(normal), DVec3::X)
    };
    let tangent_v = normalize_or(normal.cross(tangent_u), DVec3::Y);
    (tangent_u, tangent_v)
}

fn tangent_basis_for_hit(hit_ctx: &HitContext<'_>, normal: DVec3) -> (DVec3, DVec3) {
    let fallback = tangent_basis(normal);
    let Some((tangent, bitangent)) = hit_ctx.tangent_frame else {
        return fallback;
    };
    let tangent = normalize_or(tangent - normal * tangent.dot(normal), fallback.0);
    let bitangent = normalize_or(
        bitangent - normal * bitangent.dot(normal) - tangent * bitangent.dot(tangent),
        fallback.1,
    );
    (tangent, bitangent)
}

fn perlin(point: DVec3) -> f64 {
    let base = point.floor();
    let local = point - base;
    let fade = |value: f64| value * value * value * (value * (value * 6.0 - 15.0) + 10.0);
    let fade = DVec3::new(fade(local.x), fade(local.y), fade(local.z));
    let mut total = 0.0;
    for z in 0..=1 {
        for y in 0..=1 {
            for x in 0..=1 {
                let lattice = base + DVec3::new(f64::from(x), f64::from(y), f64::from(z));
                let delta = local - DVec3::new(f64::from(x), f64::from(y), f64::from(z));
                let angle = lattice_random(lattice, 0) * std::f64::consts::TAU;
                let vertical = lattice_random(lattice, 1) * 2.0 - 1.0;
                let horizontal = (1.0 - vertical * vertical).max(0.0).sqrt();
                let gradient =
                    DVec3::new(horizontal * angle.cos(), horizontal * angle.sin(), vertical);
                let wx = if x == 0 { 1.0 - fade.x } else { fade.x };
                let wy = if y == 0 { 1.0 - fade.y } else { fade.y };
                let wz = if z == 0 { 1.0 - fade.z } else { fade.z };
                total += gradient.dot(delta) * wx * wy * wz;
            }
        }
    }
    total * 1.6
}

fn lattice_random(point: DVec3, channel: i32) -> f64 {
    let mut hash = 0x811c_9dc5_u32 ^ channel as u32;
    for value in [
        point.x.floor() as i64,
        point.y.floor() as i64,
        point.z.floor() as i64,
    ] {
        hash = (hash ^ value as u32).wrapping_mul(0x0100_0193);
        hash ^= (value >> 32) as u32;
    }
    hash ^= hash >> 16;
    hash = hash.wrapping_mul(0x7feb_352d);
    hash ^= hash >> 15;
    hash = hash.wrapping_mul(0x846c_a68b);
    hash ^= hash >> 16;
    f64::from(hash) / f64::from(u32::MAX)
}

fn invalid(message: impl Into<String>) -> PotError {
    PotError::new(ErrorCode::InvalidOperation, message)
}

fn eval_error(message: impl Into<String>) -> PotError {
    PotError::new(ErrorCode::EvaluationFailed, message)
}

fn unsupported_node(id: &Id, kind: &str) -> PotError {
    PotError::with_details(
        ErrorCode::UnsupportedFeature,
        format!("shader node behavior `{kind}` is not implemented"),
        json!({"feature_id":format!("shader.node.{kind}"),"node_id":id}),
    )
}

fn unsupported_node_id(node: &str, output: &str) -> PotError {
    PotError::with_details(
        ErrorCode::UnsupportedFeature,
        format!("shader output `{output}` of `{node}` is not implemented"),
        json!({"feature_id":format!("shader.node.{node}.{output}")}),
    )
}

#[cfg(test)]
mod tests {
    #![expect(clippy::unwrap_used, reason = "tests")]

    use std::collections::BTreeMap;

    use glam::DVec3;
    use proptest::prelude::*;
    use serde_json::{Value, json};

    use crate::{
        graph::{GraphKind, GraphLink, GraphNode, NodeGroup},
        model::{Id, Material, Registry},
    };

    use super::{HitContext, color_lerp, evaluate_displacement, evaluate_surface};

    fn insert_node(group: &mut NodeGroup, id: &str, node_type: &str, inputs: Value) {
        let mut node = GraphNode::new(node_type);
        let Value::Object(inputs) = inputs else {
            unreachable!("shader test inputs must be an object")
        };
        node.inputs = inputs.into_iter().collect::<BTreeMap<_, _>>();
        group.nodes.insert(Id::new(id.to_owned()).unwrap(), node);
    }

    fn link(group: &mut NodeGroup, from: &str, output: &str, to: &str, input: &str) {
        group.links.push(GraphLink {
            from_node: Id::new(from.to_owned()).unwrap(),
            from_socket: output.to_owned(),
            to_node: Id::new(to.to_owned()).unwrap(),
            to_socket: input.to_owned(),
        });
    }

    #[test]
    fn material_volume_output_evaluates_principled_density_color_and_anisotropy() {
        let mut graph = NodeGroup::new("Volume", GraphKind::Shader);
        insert_node(&mut graph, "output", "ShaderNodeOutputMaterial", json!({}));
        insert_node(
            &mut graph,
            "volume",
            "ShaderNodeVolumePrincipled",
            json!({"Density":0.7,"Color":[0.2,0.4,0.6,1.0],"Anisotropy":0.35}),
        );
        link(&mut graph, "volume", "Volume", "output", "Volume");
        let graph_id = Id::new("volume_shader").unwrap();
        let mut groups = Registry::new();
        groups.insert(graph_id.clone(), graph);
        let material = Material {
            node_tree: Some(graph_id),
            ..Material::default()
        };
        let params = evaluate_surface(&material, &HitContext::new(&groups)).unwrap();
        assert_eq!(params.volume_density, 0.7);
        assert_eq!(params.volume_color, [0.2, 0.4, 0.6]);
        assert_eq!(params.volume_anisotropy, 0.35);
        assert!(params.volume_scattering);
    }

    #[test]
    fn displacement_output_evaluates_scale_midlevel_and_surface_normal() {
        let mut graph = NodeGroup::new("Displacement", GraphKind::Shader);
        insert_node(&mut graph, "output", "ShaderNodeOutputMaterial", json!({}));
        insert_node(
            &mut graph,
            "displacement",
            "ShaderNodeDisplacement",
            json!({"Height":0.75,"Midlevel":0.5,"Scale":2.0}),
        );
        link(
            &mut graph,
            "displacement",
            "Displacement",
            "output",
            "Displacement",
        );
        let graph_id = Id::new("displacement_shader").unwrap();
        let mut groups = Registry::new();
        groups.insert(graph_id.clone(), graph);
        let material = Material {
            node_tree: Some(graph_id),
            ..Material::default()
        };
        let mut context = HitContext::new(&groups);
        context.normal = DVec3::Z;
        let params = evaluate_surface(&material, &context).unwrap();
        assert!((params.displacement_height - 0.5).abs() < f64::EPSILON);
        assert!((evaluate_displacement(&material, &context).unwrap() - 0.5).abs() < f64::EPSILON);
    }

    proptest! {
        #[test]
        fn color_mix_matches_linear_reference(factor in 0.0_f64..=1.0) {
            let first = [0.2, 0.4, 0.6, 0.8];
            let second = [0.8, 0.6, 0.4, 0.2];
            let actual = color_lerp(first, second, factor);
            for (index, value) in actual.into_iter().enumerate() {
                let reference = first[index] + (second[index] - first[index]) * factor;
                prop_assert!((value - reference).abs() <= f64::EPSILON);
            }
        }

        #[test]
        fn color_mix_clamps_factor(factor in -1000.0_f64..=1000.0) {
            let value = color_lerp([0.0; 4], [1.0; 4], factor);
            prop_assert!(value.iter().all(|component| (0.0..=1.0).contains(component)));
        }
    }
}

#[cfg(kani)]
#[kani::proof]
fn kani_bounded_mix_factor_stays_unit_interval() {
    let factor: f64 = kani::any();
    kani::assume(factor.is_finite());
    let bounded = bounded_factor(factor);
    assert!(bounded >= 0.0 && bounded <= 1.0);
}
