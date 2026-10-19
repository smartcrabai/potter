use std::{
    collections::{BTreeMap, BTreeSet},
    fmt::Write as _,
    fs,
    path::{Path, PathBuf},
};

use glam::DVec3;
use serde_json::json;

use super::{
    ExchangeMesh, ExchangeVertexGroup, ImportedGraph, Loss, gltf, import_graph_from_meshes,
    material_surface::{
        self, AlphaMode, ChannelSource, SampledTexture, Specular, TextureTransform, TextureWrap,
    },
    obj_group_is_face_compatible, shading, single_uv_set, srgb_decode, srgb_encode,
};
use crate::{
    error::{ErrorCode, PotError, Result},
    geom::color_layers::{ColorDomain, ColorLayer, ColorStore},
    image::{ImageExtension, ImageInterpolation, load_image_data},
    model::{
        Id, Image, ImageAlphaMode, ImageColorspace, ImageSource, Material, SceneDoc, TextureRef,
    },
};

/// Blender's default OBJ axes (forward -Z, up Y): Potter's Z-up `(x, y, z)` is written as
/// `(x, z, -y)`, so Blender's default importer reads the same placement.
fn to_obj_axes(value: [f64; 3]) -> [f64; 3] {
    [value[0], value[2], -value[1]]
}

/// Inverse of `to_obj_axes`.
fn from_obj_axes(value: [f64; 3]) -> [f64; 3] {
    [value[0], -value[2], value[1]]
}

/// What an OBJ export writes: the object and MTL text, plus texture PNGs beside the MTL as
/// `(file name, bytes)`. Losses are shared through `element_data_losses` and `material_diagnostics`.
pub(crate) struct ObjExport {
    pub(crate) object: Vec<u8>,
    pub(crate) material: Vec<u8>,
    pub(crate) textures: Vec<(String, Vec<u8>)>,
}

pub(crate) fn export(
    doc: &SceneDoc,
    root_path: &Path,
    meshes: &[ExchangeMesh],
) -> Result<ObjExport> {
    let mut object = String::from("mtllib scene.mtl\n");
    let mut base = 0_usize;
    let mut normal_base = 0_usize;
    let mut uv_base = 0_usize;
    // Groups stay in effect across `o` lines, so an object without groups must reset them.
    let mut groups_on = false;
    let bound_materials = material_bindings(doc, meshes)?;
    for mesh in meshes {
        object.push_str("o ");
        object.push_str(&obj_name(&mesh.name));
        object.push('\n');
        let colors = vertex_colors(mesh)?;
        for (index, position) in mesh.positions.iter().enumerate() {
            let [x, y, z] = to_obj_axes(*position);
            let written = match colors.as_ref().and_then(|colors| colors.get(index)) {
                Some([red, green, blue]) => {
                    writeln!(object, "v {x:.9} {y:.9} {z:.9} {red} {green} {blue}")
                }
                None => writeln!(object, "v {x:.9} {y:.9} {z:.9}"),
            };
            written.map_err(|error| PotError::new(ErrorCode::InternalError, error.to_string()))?;
        }
        let (faces, normals) = shading::baked_faces(mesh)?;
        // Blender writes zero normals for singular transforms.
        let normals = if mesh.singular {
            vec![[0.0; 3]; normals.len()]
        } else {
            normals
        };
        let normals = normals.into_iter().map(to_obj_axes).collect::<Vec<_>>();
        let (normal_lines, corner_ids) = indexed_table(&normals, |normal| {
            format!("vn {:.6} {:.6} {:.6}", normal[0], normal[1], normal[2])
        });
        for line in &normal_lines {
            object.push_str(line);
            object.push('\n');
        }
        let uv_values = mesh
            .uv_active
            .and_then(|index| mesh.uv_layers.get(index))
            .map(|layer| {
                shading::baked_corners(mesh, &layer.corners)
                    .into_iter()
                    .flatten()
                    .map(|[u, v]| [u, 1.0 - v])
                    .collect::<Vec<_>>()
            });
        // Blender writes the active layer's `vt` lines in first-use order, one per distinct value.
        // Storage V points down; Blender's OBJ writes raw Blender V, which points up.
        let (uv_lines, uv_ids) = match uv_values {
            Some(values) => {
                let (lines, ids) = indexed_table(&values, |[u, v]| format!("vt {u:.6} {v:.6}"));
                (lines, Some(ids))
            }
            None => (Vec::new(), None),
        };
        for line in &uv_lines {
            object.push_str(line);
            object.push('\n');
        }
        let face_materials = (0..mesh.faces.len())
            .map(|face| bound_material(mesh, face))
            .collect::<Vec<_>>();
        let compatible_groups = mesh
            .vertex_groups
            .iter()
            .filter(|group| obj_group_is_face_compatible(mesh, group))
            .map(|group| {
                (
                    group.name.as_str(),
                    group
                        .weights
                        .iter()
                        .map(|(vertex, _)| *vertex)
                        .collect::<BTreeSet<_>>(),
                )
            })
            .collect::<Vec<_>>();
        // Blender uses an empty material name to clear the current material.
        object.push_str("usemtl\n");
        if groups_on {
            object.push_str("g\n");
        }
        let mut material_state = Some(None::<&Id>);
        let mut group_state = None::<&str>;
        let mut smooth_state = None::<bool>;
        let mut corner = 0_usize;
        for (face_index, (face, is_smooth)) in
            faces.iter().zip(mesh.smooth.iter().copied()).enumerate()
        {
            let first_corner = corner;
            corner = first_corner.checked_add(face.len()).ok_or_else(|| {
                PotError::new(ErrorCode::ExportFailed, "OBJ corner count overflow")
            })?;
            if face.len() < 3 {
                continue;
            }
            let material = face_materials.get(face_index).copied().flatten();
            if material_state != Some(material) {
                if let Some(id) = material {
                    let name = bound_materials.get(id).ok_or_else(|| {
                        PotError::new(ErrorCode::InternalError, "OBJ material binding is missing")
                    })?;
                    writeln!(object, "usemtl {name}").map_err(|error| {
                        PotError::new(ErrorCode::InternalError, error.to_string())
                    })?;
                } else {
                    object.push_str("usemtl\n");
                }
                material_state = Some(material);
            }
            let mut face_group = None;
            for (name, membership) in &compatible_groups {
                if face.iter().all(|vertex| membership.contains(vertex)) && face_group.is_none() {
                    face_group = Some(*name);
                }
            }
            if face_group != group_state {
                object.push('g');
                if let Some(name) = face_group {
                    write!(object, " {name}").map_err(|error| {
                        PotError::new(ErrorCode::InternalError, error.to_string())
                    })?;
                }
                object.push('\n');
                group_state = face_group;
            }
            if smooth_state != Some(is_smooth) {
                writeln!(object, "s {}", u8::from(is_smooth))
                    .map_err(|error| PotError::new(ErrorCode::InternalError, error.to_string()))?;
                smooth_state = Some(is_smooth);
            }
            object.push('f');
            for (offset, index) in face.iter().enumerate() {
                let vertex = base
                    .checked_add(*index)
                    .and_then(|index| index.checked_add(1))
                    .ok_or_else(|| {
                        PotError::new(ErrorCode::ExportFailed, "OBJ vertex index overflow")
                    })?;
                write!(object, " {vertex}")
                    .map_err(|error| PotError::new(ErrorCode::InternalError, error.to_string()))?;
                let normal = first_corner
                    .checked_add(offset)
                    .and_then(|corner| corner_ids.get(corner))
                    .and_then(|local| normal_base.checked_add(*local))
                    .and_then(|index| index.checked_add(1))
                    .ok_or_else(|| {
                        PotError::new(
                            ErrorCode::ExportFailed,
                            "OBJ normal count does not match faces",
                        )
                    })?;
                let uv = match &uv_ids {
                    Some(ids) => first_corner
                        .checked_add(offset)
                        .and_then(|corner| ids.get(corner))
                        .and_then(|local| uv_base.checked_add(*local))
                        .and_then(|index| index.checked_add(1))
                        .map(|index| index.to_string())
                        .ok_or_else(|| {
                            PotError::new(
                                ErrorCode::ExportFailed,
                                "OBJ UV count does not match faces",
                            )
                        })?,
                    None => String::new(),
                };
                write!(object, "/{uv}/{normal}")
                    .map_err(|error| PotError::new(ErrorCode::InternalError, error.to_string()))?;
            }
            object.push('\n');
        }
        groups_on = group_state.is_some();
        base = base
            .checked_add(mesh.positions.len())
            .ok_or_else(|| PotError::new(ErrorCode::ExportFailed, "OBJ vertex index overflow"))?;
        normal_base = normal_base
            .checked_add(normal_lines.len())
            .ok_or_else(|| PotError::new(ErrorCode::ExportFailed, "OBJ normal index overflow"))?;
        uv_base = uv_base
            .checked_add(uv_lines.len())
            .ok_or_else(|| PotError::new(ErrorCode::ExportFailed, "OBJ UV index overflow"))?;
    }
    let mtl = mtl_entries(doc, root_path, &bound_materials, true)?;
    Ok(ObjExport {
        object: object.into_bytes(),
        material: mtl.text.into_bytes(),
        textures: mtl.textures,
    })
}

struct MtlEntries {
    text: String,
    textures: Vec<(String, Vec<u8>)>,
    losses: Vec<Loss>,
}

/// Names each bound material for the OBJ `usemtl` and MTL `newmtl` lines; see the rule below.
fn material_bindings(doc: &SceneDoc, meshes: &[ExchangeMesh]) -> Result<BTreeMap<Id, String>> {
    let mut bound = BTreeSet::<Id>::new();
    for mesh in meshes {
        if !mesh.face_material_indices.is_empty()
            && mesh.face_material_indices.len() != mesh.faces.len()
        {
            return Err(PotError::new(
                ErrorCode::ExportFailed,
                "OBJ face material slots do not match the face count",
            ));
        }
        for (face, geometry) in mesh.faces.iter().enumerate() {
            if geometry.len() >= 3
                && let Some(id) = bound_material(mesh, face)
            {
                bound.insert(id.clone());
            }
        }
    }
    // Blender names each material by `name.replace(' ', "_")` (`obj_export_mtl.cc`). A name that is
    // empty or holds other whitespace or `#` (an MTL comment), or that an earlier material already
    // took, falls back to the material's ID. Valid names are claimed first, so a fallback never
    // steals a display name; every final name goes through one used-names set.
    let mut used = BTreeSet::<String>::new();
    let mut names = BTreeMap::<Id, String>::new();
    let mut fallbacks = Vec::<Id>::new();
    for id in &bound {
        let candidate = doc
            .materials
            .get(id)
            .map(|material| material.name.replace(' ', "_"))
            .unwrap_or_default();
        if !candidate.is_empty()
            && !candidate.contains(char::is_whitespace)
            && !candidate.contains('#')
            && used.insert(candidate.clone())
        {
            names.insert(id.clone(), candidate);
        } else {
            fallbacks.push(id.clone());
        }
    }
    for id in fallbacks {
        let root = id.as_str().to_owned();
        let mut name = root.clone();
        let mut index = 1_u32;
        while used.contains(&name) {
            name = format!("{root}_{index}");
            index += 1;
        }
        used.insert(name.clone());
        names.insert(id, name);
    }
    Ok(names)
}

fn mtl_entries(
    doc: &SceneDoc,
    root_path: &Path,
    bindings: &BTreeMap<Id, String>,
    write_output: bool,
) -> Result<MtlEntries> {
    let mut output = MtlEntries {
        text: String::new(),
        textures: Vec::new(),
        losses: Vec::new(),
    };
    for (id, name) in bindings {
        let mut entry = MtlWriter {
            doc,
            root_path,
            id,
            name: name.as_str(),
            textures: Vec::new(),
            losses: Vec::new(),
            write_output,
        };
        output.text.push_str(&entry.material_mtl()?);
        output.textures.append(&mut entry.textures);
        output.losses.append(&mut entry.losses);
    }
    Ok(output)
}

/// MTL losses for bound materials without writing MTL text or loading texture pixels.
pub(crate) fn material_diagnostics(
    doc: &SceneDoc,
    root_path: &Path,
    meshes: &[ExchangeMesh],
) -> Result<Vec<Loss>> {
    let bindings = material_bindings(doc, meshes)?;
    Ok(mtl_entries(doc, root_path, &bindings, false)?.losses)
}

/// One bound material's MTL entry under construction: its texture PNGs and the losses found.
struct MtlWriter<'a> {
    doc: &'a SceneDoc,
    root_path: &'a Path,
    id: &'a Id,
    name: &'a str,
    textures: Vec<(String, Vec<u8>)>,
    losses: Vec<Loss>,
    write_output: bool,
}

impl MtlWriter<'_> {
    fn loss(&mut self, feature: &str, reason: &str) {
        if !self.write_output {
            self.losses.push(Loss {
                feature_id: feature.to_owned(),
                data_id: Some(self.id.as_str().to_owned()),
                reason: reason.to_owned(),
                suggestion: None,
            });
        }
    }

    /// Whether the base colour image carries alpha below 1 in any texel. A source without an alpha
    /// channel, or with opaque texels, has no texture alpha to lose: Blender reads its Alpha as 1.
    /// If the image cannot be loaded, its alpha is unknown and is treated as present.
    fn texture_has_alpha(&self, texture: &SampledTexture) -> bool {
        let Some(source) = texture.channels[3].as_ref() else {
            return false;
        };
        let Some(image) = self.doc.images.get(&source.image) else {
            return true;
        };
        match load_image_data(image, self.root_path, texture.interpolation) {
            Ok(data) => data
                .pixels
                .iter()
                .chain(data.tiles.values().flat_map(|tile| tile.pixels.iter()))
                .any(|pixel| pixel[3] < 1.0),
            Err(_) => true,
        }
    }

    /// One `map_*` line. Its options are Blender's `-o`, `-s`, and `-bm`; its file is the texture
    /// packed from `channels` and written as PNG beside the MTL.
    fn map(
        &mut self,
        key: &str,
        texture: &SampledTexture,
        channels: &[Option<ChannelSource>; 4],
        bump: Option<f64>,
    ) -> Result<String> {
        let mut options = String::new();
        if let Some(transform) = &texture.transform {
            if transform.rotation != 0.0 {
                self.loss(
                    "material.mtl.texture_rotation",
                    "MTL has no texture rotation option; the rotation is not written",
                );
            }
            if self.write_output {
                let (location, scale) = mtl_transform(transform);
                #[expect(
                    clippy::float_cmp,
                    reason = "exact zero test: only an exactly zero location is omitted from -o"
                )]
                if location != [0.0; 3] {
                    let [x, y, z] = location;
                    let _ = write!(options, " -o {x:.6} {y:.6} {z:.6}");
                }
                #[expect(
                    clippy::float_cmp,
                    reason = "exact unit test: only an exactly unit scale is omitted from -s"
                )]
                if scale != [1.0; 3] {
                    let [x, y, z] = scale;
                    let _ = write!(options, " -s {x:.6} {y:.6} {z:.6}");
                }
            }
        }
        if texture.wrap != [TextureWrap::Repeat; 2] {
            self.loss(
                "material.mtl.texture_wrap",
                "MTL has no texture wrap option; the wrap mode is not written",
            );
        }
        if self.write_output
            && let Some(strength) = bump
            && strength > 0.0001
        {
            let _ = write!(options, " -bm {strength:.6}");
        }
        if !self.write_output {
            return Ok(String::new());
        }
        let file = format!("{}_{}.png", self.name, key.trim_start_matches("map_"));
        let packed =
            gltf::pack_channels(channels, texture.interpolation, self.doc, self.root_path)?;
        let bytes = gltf::encode_png_with_colorspace(ImageColorspace::Linear, &packed.pixels)?;
        self.textures.push((file.clone(), bytes));
        Ok(format!("{key}{options} {file}\n"))
    }

    /// The MTL entry of this material, from the resolved surface with Blender's non-PBR rules
    /// (`store_bsdf_properties` and `MTLWriter::write_bsdf_properties`, `obj_export_mtl.cc` and
    /// `obj_export_file_writer.cc`). A map replaces its scalar line, and the maps follow the scalars
    /// in Blender's `MTLTexMapType` order.
    fn material_mtl(&mut self) -> Result<String> {
        let (surface, surface_losses) = material_surface::resolve_surface(self.doc, self.id)?;
        if !self.write_output {
            self.losses.extend(surface_losses);
        }
        if !surface.double_sided {
            self.loss(
                "material.mtl.double_sided",
                "MTL has no two-sided flag; Blender's OBJ importer reads the material as double-sided",
            );
        }
        let [red, green, blue, alpha] = surface.base_color.factor;
        let metallic = surface.metallic;
        let level = surface.specular.as_ref().map_or(0.5, specular_level);
        let open = 1.0 - surface.roughness;
        // Blender's empirical inverse of the importer's `roughness = 1 - sqrt(Ns / 1000)`.
        let ns = open * (open * 1000.0);
        #[expect(
            clippy::float_cmp,
            reason = "exact opacity test: only an alpha of exactly 1.0 is opaque"
        )]
        let illum = illum_mode(level, metallic, alpha != 1.0);

        // Blender's `map_d` names the image linked to Alpha. When the colour image's alpha is the
        // material alpha, that is the colour image itself, and Potter's importer reads alpha only
        // from the image `map_Kd` names. The colour image therefore keeps its alpha channel and
        // `map_d` names it. A material with a scalar alpha factor keeps `d`.
        let texture_alpha = !matches!(surface.alpha_mode, AlphaMode::Opaque)
            && surface
                .base_color
                .texture
                .as_ref()
                .is_some_and(|texture| self.texture_has_alpha(texture));
        #[expect(
            clippy::float_cmp,
            reason = "exact unit test: only a factor alpha of exactly 1.0 leaves the texture alpha as the material alpha"
        )]
        let alpha_from_map = texture_alpha && alpha == 1.0;
        if texture_alpha && !alpha_from_map {
            // `map_d` would carry this alpha, but Potter's importer reads a map's alpha only when
            // the factor is 1, so the texture alpha is not written.
            self.loss(
                "material.mtl.alpha_texture",
                "MTL writes the colour image without its alpha when the alpha factor is not 1; the texture alpha is not written",
            );
        }
        let kd_map = match &surface.base_color.texture {
            Some(texture) => {
                let channels = if alpha_from_map {
                    texture.channels.clone()
                } else {
                    rgb_channels(&texture.channels)
                };
                Some(self.map("map_Kd", texture, &channels, None)?)
            }
            None => None,
        };
        let ks_map = match surface
            .specular
            .as_ref()
            .and_then(|specular| specular.texture.as_ref())
        {
            Some(texture) if texture.channels[3].is_some() => Some(self.map(
                "map_Ks",
                texture,
                &gray_channels(texture.channels[3].clone()),
                None,
            )?),
            _ => None,
        };
        let ns_map = match &surface.metallic_roughness_texture {
            Some(texture) if texture.channels[1].is_some() => Some(self.map(
                "map_Ns",
                texture,
                &gray_channels(texture.channels[1].clone()),
                None,
            )?),
            _ => None,
        };
        if surface
            .metallic_roughness_texture
            .as_ref()
            .is_some_and(|texture| texture.channels[2].is_some())
        {
            self.loss(
                "material.mtl.metallic_texture",
                "Blender's non-PBR MTL output omits the metallic map; only the roughness map is written",
            );
        }
        #[expect(
            clippy::float_cmp,
            reason = "exact zero test: an emissive factor of exactly zero skips map_Ke, as Blender does for zero strength"
        )]
        let ke_map = match &surface.emissive.texture {
            // Blender skips map_Ke when Emission Strength is zero.
            // shortcut: Potter keeps only the emitted factor (colour times strength), so a textured
            // emission with black colour and nonzero strength is not written, unlike Blender.
            Some(texture) if surface.emissive.factor != [0.0; 3] => {
                Some(self.map("map_Ke", texture, &rgb_channels(&texture.channels), None)?)
            }
            _ => None,
        };
        let d_map = kd_map
            .as_ref()
            .filter(|_| alpha_from_map)
            .map(|line| line.replacen("map_Kd", "map_d", 1));
        let bump_map = match &surface.normal {
            Some(normal) => Some(self.map(
                "map_Bump",
                &normal.texture,
                &normal.texture.channels,
                Some(normal.scale),
            )?),
            None => None,
        };

        if surface.clearcoat.is_some() {
            self.loss(
                "material.mtl.clearcoat",
                "Blender writes clearcoat only in PBR mode; the non-PBR MTL has no clearcoat",
            );
        }
        if surface.sheen.is_some() {
            self.loss(
                "material.mtl.sheen",
                "Blender writes sheen only in PBR mode; the non-PBR MTL has no sheen",
            );
        }
        if surface.anisotropy.is_some() {
            self.loss(
                "material.mtl.anisotropy",
                "Blender writes anisotropy only in PBR mode; the non-PBR MTL has no anisotropy",
            );
        }
        if surface.transmission.is_some() {
            self.loss(
                "material.mtl.transmission",
                "Blender writes transmission only in PBR mode; the non-PBR MTL has no transmission",
            );
        }
        if surface.occlusion.is_some() {
            self.loss(
                "material.mtl.occlusion",
                "MTL has no occlusion map; the occlusion texture is not written",
            );
        }
        if surface.unlit {
            self.loss(
                "material.mtl.unlit",
                "MTL has no unlit model; the material is written as a lit Principled surface",
            );
        }
        if matches!(surface.alpha_mode, AlphaMode::Mask { .. }) {
            self.loss(
                "material.mtl.alpha_cutoff",
                "MTL has no alpha cutoff; the alpha is written as d",
            );
        }
        if let Some(specular) = &surface.specular {
            // Blender's tint is white; a textured or tinted specular has no MTL form.
            let expected = (2.0 * level).max(1.0);
            if specular.color_texture.is_some()
                || specular
                    .color_factor
                    .iter()
                    .any(|channel| (*channel - expected).abs() > 1.0e-9)
            {
                self.loss(
                    "material.mtl.specular_tint",
                    "MTL has no specular tint; only the specular level is written",
                );
            }
        }

        if !self.write_output {
            return Ok(String::new());
        }
        let mut text = format!("newmtl {}\n", self.name);
        if ns_map.is_none() {
            let _ = writeln!(text, "Ns {ns:.6}");
        }
        let ka = if metallic == 0.0 { 1.0 } else { metallic };
        let _ = writeln!(text, "Ka {ka:.6} {ka:.6} {ka:.6}");
        if kd_map.is_none() {
            let _ = writeln!(text, "Kd {red:.6} {green:.6} {blue:.6}");
        }
        if ks_map.is_none() {
            let _ = writeln!(text, "Ks {level:.6} {level:.6} {level:.6}");
        }
        if ke_map.is_none() {
            let [r, g, b] = surface.emissive.factor;
            let _ = writeln!(text, "Ke {r:.6} {g:.6} {b:.6}");
        }
        let _ = writeln!(text, "Ni {:.6}", surface.ior);
        if d_map.is_none() {
            let _ = writeln!(text, "d {alpha:.6}");
        }
        let _ = writeln!(text, "illum {illum}");
        for line in [kd_map, ks_map, ns_map, ke_map, d_map, bump_map]
            .into_iter()
            .flatten()
        {
            text.push_str(&line);
        }
        Ok(text)
    }
}

/// Blender's specular level from a Potter specular input. The exporter splits `2 * level` into
/// `factor = min(2 * level, 1)` and a tint of `max(2 * level, 1)`, so the level is recovered from
/// whichever holds it. `shortcut: assumes a white tint, which the MTL writer checks separately.`
fn specular_level(specular: &Specular) -> f64 {
    if specular.factor < 1.0 {
        specular.factor / 2.0
    } else {
        mean3(specular.color_factor) / 2.0
    }
}

/// Blender's `illum` for a material: 1 without specular, 3 for metal and 6 when that metal is
/// transparent, 9 for transparent dielectrics, and 2 otherwise (`store_bsdf_properties`).
fn illum_mode(specular: f64, metallic: f64, transparent: bool) -> u8 {
    if specular == 0.0 {
        1
    } else if metallic > 0.0 {
        if transparent { 6 } else { 3 }
    } else if transparent {
        9
    } else {
        2
    }
}

/// Blender's `-o` location and `-s` scale for a texture transform: the inverse of the glTF
/// transform built from a Blender point mapping (`point_transform` in `material_surface.rs`). Rotation
/// has no MTL option, so the caller reports it.
fn mtl_transform(transform: &TextureTransform) -> ([f64; 3], [f64; 3]) {
    let (sin, cos) = transform.rotation.sin_cos();
    let [scale_x, scale_y] = transform.scale;
    let location = [
        transform.offset[0] + scale_y * sin,
        1.0 - transform.offset[1] - scale_y * cos,
        0.0,
    ];
    (location, [scale_x, scale_y, 1.0])
}

/// Colour channels of a texture, with alpha unread (MTL maps are RGB).
fn rgb_channels(channels: &[Option<ChannelSource>; 4]) -> [Option<ChannelSource>; 4] {
    [
        channels[0].clone(),
        channels[1].clone(),
        channels[2].clone(),
        None,
    ]
}

/// One source channel as grey RGB.
fn gray_channels(source: Option<ChannelSource>) -> [Option<ChannelSource>; 4] {
    [source.clone(), source.clone(), source, None]
}

fn mean3(values: [f64; 3]) -> f64 {
    (values[0] + values[1] + values[2]) / 3.0
}

/// The active color layer as nonnegative sRGB `r g b` values for OBJ's `v x y z r g b` form.
/// Alpha is not written: xyzrgb carries RGB only.
fn vertex_colors(mesh: &ExchangeMesh) -> Result<Option<Vec<[f64; 3]>>> {
    let Some(layer) = mesh
        .colors
        .active
        .as_deref()
        .and_then(|name| mesh.colors.index_of(name))
        .and_then(|index| mesh.colors.layers.get(index))
    else {
        return Ok(None);
    };
    let colors = super::point_color_values(mesh, layer)?;
    if colors.iter().flatten().any(|value| !value.is_finite()) {
        return Err(PotError::new(
            ErrorCode::ExportFailed,
            "color layer contains a non-finite value",
        ));
    }
    Ok(Some(
        colors
            .into_iter()
            .map(|[red, green, blue, _]| {
                [
                    srgb_encode(red.max(0.0)),
                    srgb_encode(green.max(0.0)),
                    srgb_encode(blue.max(0.0)),
                ]
            })
            .collect(),
    ))
}

/// The material slot a face is bound to, or `None` when it is unassigned. An index past the last
/// slot clamps to it, as on import.
fn bound_material(mesh: &ExchangeMesh, face: usize) -> Option<&Id> {
    let last = mesh.material_slots.len().checked_sub(1)?;
    let index = usize::try_from(*mesh.face_material_indices.get(face)?)
        .unwrap_or(usize::MAX)
        .min(last);
    mesh.material_slots.get(index)?.as_ref()
}

/// Blender's indexed table for one object: unique entries in first-use order, as text lines, plus
/// each value's 0-based index into those lines.
fn indexed_table<T>(values: &[T], line: impl Fn(&T) -> String) -> (Vec<String>, Vec<usize>) {
    let mut lines = Vec::<String>::new();
    let mut lookup = BTreeMap::<String, usize>::new();
    let mut corner_ids = Vec::with_capacity(values.len());
    for value in values {
        let text = line(value);
        let index = *lookup.entry(text.clone()).or_insert_with(|| {
            lines.push(text);
            lines.len() - 1
        });
        corner_ids.push(index);
    }
    (lines, corner_ids)
}

/// An object with no geometry yet. Fields an OBJ file does not carry stay empty.
fn empty_mesh(id: String, name: String) -> ExchangeMesh {
    ExchangeMesh {
        id,
        name,
        positions: Vec::new(),
        faces: Vec::new(),
        normals: None,
        mirrored: false,
        singular: false,
        vertex_normals: Vec::new(),
        smooth: Vec::new(),
        uv_layers: Vec::new(),
        uv_active: None,
        uv_render: None,
        colors: ColorStore::default(),
        vertex_groups: Vec::new(),
        material_slots: Vec::new(),
        face_material_indices: Vec::new(),
    }
}

/// What an OBJ file reads to: its objects, and the MTL materials of every library it names.
struct ObjImport {
    meshes: Vec<ExchangeMesh>,
    losses: Vec<Loss>,
    materials: BTreeMap<String, MtlMaterial>,
    external_material_names: BTreeMap<Id, String>,
}

fn import(path: &Path) -> Result<ObjImport> {
    let text = fs::read_to_string(path).map_err(|error| PotError::io(&error))?;
    let mut positions = Vec::<[f64; 3]>::new();
    // Linear color of each `v` line, in file order (`v` indices are global).
    let mut position_colors = Vec::<Option<[f64; 4]>>::new();
    let mut normals = Vec::<DVec3>::new();
    let mut meshes = Vec::<ExchangeMesh>::new();
    let mut losses = Vec::<Loss>::new();
    let mut current = empty_mesh("object_0".to_owned(), "Object".to_owned());
    // Normal named by each face corner of `current`, in face order.
    let mut corner_normals = Vec::<Option<DVec3>>::new();
    // Storage UV of each `vt` line, in file order (`vt` indices are global), and the UV named by
    // each face corner of `current`, in face order.
    let mut uvs = Vec::<[f64; 2]>::new();
    let mut corner_uvs = Vec::<Option<[f64; 2]>>::new();
    // Linear color of each local vertex of `current`, in local order.
    let mut local_colors = Vec::<Option<[f64; 4]>>::new();
    // Groups named by the `g` line in effect, and the local vertices each group holds.
    let mut active_groups = Vec::<String>::new();
    let mut group_members = BTreeMap::<String, BTreeSet<usize>>::new();
    // Every `usemtl` name read, and the one in effect for each face of `current`, in face order.
    let mut material_names = Vec::<String>::new();
    let mut imported_material_ids = BTreeMap::<String, Id>::new();
    let mut external_material_names = BTreeMap::<Id, String>::new();
    let mut active_material = None::<usize>;
    let mut face_materials = Vec::<Option<usize>>::new();
    // Whether an `s` group is on at the current line, and whether any face of `current` was read
    // while one was on.
    let mut group_on = false;
    let mut smoothed = false;
    let mut remap = BTreeMap::<usize, usize>::new();
    // Names of every object declared by `o`. Objects without geometry are dropped from a file
    // that has geometry; a file with no geometry imports each declared object empty, so
    // Potter's own empty-object exports re-read.
    let mut declared = Vec::<String>::new();
    // Libraries named by `mtllib`, in file order; the last non-empty `o` name; faces read so far.
    let mut libraries = Vec::<String>::new();
    let mut last_object = None::<String>;
    let mut face_count = 0_usize;
    for (line_number, line) in text.lines().enumerate() {
        let mut words = line.split_whitespace();
        let Some(kind) = words.next() else {
            continue;
        };
        match kind {
            "v" => {
                let (position, color) = parse_vertex(words, line_number)?;
                positions.push(from_obj_axes(position));
                position_colors.push(color);
            }
            "vn" => {
                let values = parse_floats(words, 3, line_number)?;
                // Zero normals are valid: Blender writes them for degenerate faces and singular nodes.
                let normal = DVec3::from_array(from_obj_axes([values[0], values[1], values[2]]));
                if !normal.is_finite() {
                    return Err(parse_error(line_number));
                }
                normals.push(normal);
            }
            "vt" => {
                // Blender's OBJ writes raw V, which points up; Potter stores V down. The OBJ format
                // makes `v` optional with a default of 0, so `vt u` is a complete coordinate.
                let u = parse_floats(words.by_ref(), 1, line_number)?[0];
                let v = if words.clone().next().is_some() {
                    parse_floats(words, 1, line_number)?[0]
                } else {
                    0.0
                };
                uvs.push([u, 1.0 - v]);
            }
            "o" => {
                let name = words.collect::<Vec<_>>().join(" ");
                declared.push(name.clone());
                if !name.is_empty() {
                    last_object = Some(name.clone());
                }
                if !current.faces.is_empty() || !current.positions.is_empty() {
                    finish_object(
                        &mut current,
                        &corner_normals,
                        &corner_uvs,
                        &group_members,
                        smoothed,
                        &mut losses,
                    );
                    finish_colors(&mut current, &local_colors, &mut losses);
                    finish_materials(
                        &mut current,
                        &face_materials,
                        &material_names,
                        &mut imported_material_ids,
                        &mut external_material_names,
                    )?;
                    meshes.push(current);
                    current = empty_mesh(
                        format!("object_{}", meshes.len()),
                        if name.is_empty() {
                            format!("Object {}", meshes.len())
                        } else {
                            name
                        },
                    );
                    corner_normals.clear();
                    corner_uvs.clear();
                    local_colors.clear();
                    group_members.clear();
                    face_materials.clear();
                    smoothed = false;
                    remap.clear();
                } else if !name.is_empty() {
                    current.name = name;
                }
            }
            "g" => {
                // Blender treats the entire remainder as one group name (`geom_update_group`).
                let name = split_word(line).1.trim();
                active_groups = if name.is_empty()
                    || name.contains("off")
                    || name.contains("null")
                    || name.contains("default")
                {
                    Vec::new()
                } else {
                    vec![name.to_owned()]
                };
            }
            "mtllib" => {
                let name = words.collect::<Vec<_>>().join(" ");
                // Blender strips quotes only when both ends are quoted (`add_mtl_library`).
                let unquoted = name.len() > 2 && name.starts_with('"') && name.ends_with('"');
                let name = if unquoted {
                    name[1..name.len() - 1].to_owned()
                } else {
                    name
                };
                if !libraries.contains(&name) {
                    libraries.push(name);
                }
            }
            "usemtl" => {
                let name = split_word(line).1.trim().to_owned();
                active_material = (!name.is_empty()).then(|| {
                    if let Some(index) = material_names.iter().position(|known| *known == name) {
                        index
                    } else {
                        material_names.push(name);
                        material_names.len() - 1
                    }
                });
            }
            "s" => {
                group_on = words
                    .next()
                    .is_some_and(|group| group != "0" && group != "off");
            }
            "f" => {
                let mut face = Vec::new();
                let mut face_normals = Vec::new();
                let mut face_uvs = Vec::new();
                for token in words {
                    let raw_index = token
                        .split('/')
                        .next()
                        .ok_or_else(|| parse_error(line_number))?;
                    let source_index = resolve_index(raw_index, positions.len(), line_number)?;
                    let position = *positions
                        .get(source_index)
                        .ok_or_else(|| parse_error(line_number))?;
                    let local_index = *remap.entry(source_index).or_insert_with(|| {
                        current.positions.push(position);
                        local_colors.push(position_colors.get(source_index).copied().flatten());
                        current.positions.len() - 1
                    });
                    face.push(local_index);
                    let normal = match token.split('/').nth(2).filter(|value| !value.is_empty()) {
                        Some(value) => {
                            let index = resolve_index(value, normals.len(), line_number)?;
                            Some(*normals.get(index).ok_or_else(|| parse_error(line_number))?)
                        }
                        None => None,
                    };
                    face_normals.push(normal);
                    let uv = match token.split('/').nth(1).filter(|value| !value.is_empty()) {
                        Some(value) => {
                            let index = resolve_index(value, uvs.len(), line_number)?;
                            Some(*uvs.get(index).ok_or_else(|| parse_error(line_number))?)
                        }
                        None => None,
                    };
                    face_uvs.push(uv);
                }
                if face.len() >= 3 {
                    for group in &active_groups {
                        group_members
                            .entry(group.clone())
                            .or_default()
                            .extend(face.iter().copied());
                    }
                    current.faces.push(face);
                    face_materials.push(active_material);
                    corner_normals.extend(face_normals);
                    corner_uvs.extend(face_uvs);
                    smoothed |= group_on;
                    face_count += 1;
                }
            }
            _ => {}
        }
    }
    if !current.faces.is_empty() || !current.positions.is_empty() {
        finish_object(
            &mut current,
            &corner_normals,
            &corner_uvs,
            &group_members,
            smoothed,
            &mut losses,
        );
        finish_colors(&mut current, &local_colors, &mut losses);
        finish_materials(
            &mut current,
            &face_materials,
            &material_names,
            &mut imported_material_ids,
            &mut external_material_names,
        )?;
        meshes.push(current);
    }
    if face_count == 0 && !positions.is_empty() {
        // Blender's `use_all_vertices_if_no_faces`: with no faces anywhere, every `v` becomes one
        // object, named by the last `o` or else by the file (`OBJParser::parse`).
        let name = last_object.unwrap_or_else(|| file_stem(path));
        let mut mesh = empty_mesh("object_0".to_owned(), name);
        mesh.positions = positions;
        finish_colors(&mut mesh, &position_colors, &mut losses);
        meshes = vec![mesh];
    } else if meshes.is_empty() {
        if declared.is_empty() {
            return Err(PotError::with_details(
                ErrorCode::ImportFailed,
                "OBJ contains no faces",
                json!({ "path": path.display().to_string() }),
            ));
        }
        let has_uvs = !uvs.is_empty();
        meshes = declared
            .into_iter()
            .enumerate()
            .map(|(index, name)| {
                let mut mesh = empty_mesh(
                    format!("object_{index}"),
                    if name.is_empty() {
                        format!("Object {index}")
                    } else {
                        name
                    },
                );
                if has_uvs {
                    (mesh.uv_layers, mesh.uv_active, mesh.uv_render) = single_uv_set(Vec::new());
                }
                mesh
            })
            .collect();
    }
    // The `.mtl` beside the `.obj` is read as well when it exists (`add_default_mtl_library`).
    let default_library = path.with_extension("mtl");
    if default_library.is_file()
        && let Some(name) = default_library
            .file_name()
            .map(|name| name.to_string_lossy().into_owned())
        && !libraries.contains(&name)
    {
        libraries.push(name);
    }
    let dir = path.parent().unwrap_or_else(|| Path::new(""));
    let mut materials = BTreeMap::new();
    for name in &libraries {
        let library_path = dir.join(name);
        match fs::read(&library_path) {
            Ok(bytes) => parse_mtl(
                &String::from_utf8_lossy(&bytes),
                library_path.parent().unwrap_or(dir),
                &mut materials,
            ),
            Err(error) => losses.push(Loss {
                feature_id: "material.library".to_owned(),
                data_id: Some(name.clone()),
                reason: format!("OBJ MTL library `{name}` cannot be read: {error}"),
                suggestion: Some("place the .mtl file beside the .obj file".to_owned()),
            }),
        }
    }
    Ok(ObjImport {
        meshes,
        losses,
        materials,
        external_material_names,
    })
}

fn file_stem(path: &Path) -> String {
    path.file_stem().map_or_else(
        || "Object".to_owned(),
        |stem| stem.to_string_lossy().into_owned(),
    )
}

/// Resolves an OBJ index. Positive values count from 1; negative values count back from the end
/// of the `count` entries read so far.
fn resolve_index(token: &str, count: usize, line: usize) -> Result<usize> {
    let index = token.parse::<i64>().map_err(|_| parse_error(line))?;
    match index.cmp(&0) {
        std::cmp::Ordering::Greater => usize::try_from(index - 1).map_err(|_| parse_error(line)),
        std::cmp::Ordering::Less => count
            .checked_sub(usize::try_from(index.unsigned_abs()).map_err(|_| parse_error(line))?)
            .ok_or_else(|| parse_error(line)),
        std::cmp::Ordering::Equal => Err(parse_error(line)),
    }
}

/// Sets complete custom normals and UV layers, and imports each OBJ group as unit-weight vertex
/// memberships. Smooth groups without any normals are reported as losses.
fn finish_object(
    mesh: &mut ExchangeMesh,
    corner_normals: &[Option<DVec3>],
    corner_uvs: &[Option<[f64; 2]>],
    group_members: &BTreeMap<String, BTreeSet<usize>>,
    smoothed: bool,
    losses: &mut Vec<Loss>,
) {
    for (name, vertices) in group_members {
        mesh.vertex_groups.push(ExchangeVertexGroup {
            name: name.clone(),
            weights: vertices.iter().map(|&vertex| (vertex, 1.0)).collect(),
        });
    }
    let supplied = corner_normals.iter().flatten().count();
    if supplied > 0 && supplied == corner_normals.len() {
        mesh.normals = Some(
            corner_normals
                .iter()
                .flatten()
                .map(DVec3::to_array)
                .collect(),
        );
    } else if supplied > 0 {
        losses.push(Loss {
            feature_id: "mesh.normals".to_owned(),
            data_id: Some(mesh.name.clone()),
            reason: "some face corners have no vn normal; partial normals are ignored and shading is recomputed".to_owned(),
            suggestion: Some("write a vn index on every face corner (f v//vn) to keep custom normals".to_owned()),
        });
    } else if smoothed {
        losses.push(Loss {
            feature_id: "mesh.smoothing".to_owned(),
            data_id: Some(mesh.name.clone()),
            reason: "smooth groups without vn normals are not mapped to shading".to_owned(),
            suggestion: Some("write vn normals (f v//vn) to keep smooth shading".to_owned()),
        });
    }
    let supplied_uvs = corner_uvs.iter().flatten().count();
    if supplied_uvs > 0 && supplied_uvs == corner_uvs.len() {
        let mut values = corner_uvs.iter().flatten().copied();
        let corners = mesh
            .faces
            .iter()
            .map(|face| values.by_ref().take(face.len()).collect::<Vec<_>>())
            .collect();
        (mesh.uv_layers, mesh.uv_active, mesh.uv_render) = single_uv_set(corners);
    } else if supplied_uvs > 0 {
        losses.push(Loss {
            feature_id: "mesh.uv_layer".to_owned(),
            data_id: Some(mesh.name.clone()),
            reason: "some face corners have no vt coordinate; partial UVs are ignored".to_owned(),
            suggestion: Some(
                "write a vt index on every face corner (f v/vt) to keep UV coordinates".to_owned(),
            ),
        });
    }
}

/// Sets the `Color` layer from the `v` colors when every vertex has one. A partial set is a loss.
fn finish_colors(
    mesh: &mut ExchangeMesh,
    local_colors: &[Option<[f64; 4]>],
    losses: &mut Vec<Loss>,
) {
    let supplied = local_colors.iter().flatten().count();
    if supplied > 0 && supplied == local_colors.len() {
        mesh.colors = ColorStore {
            layers: vec![ColorLayer {
                name: "Color".to_owned(),
                domain: ColorDomain::Point,
                values: local_colors.iter().flatten().copied().collect(),
            }],
            active: Some("Color".to_owned()),
            render: Some("Color".to_owned()),
        };
    } else if supplied > 0 {
        losses.push(Loss {
            feature_id: "mesh.color_attribute".to_owned(),
            data_id: Some(mesh.name.clone()),
            reason: "some OBJ vertices have no xyzrgb color; partial vertex colors are ignored"
                .to_owned(),
            suggestion: Some(
                "write three sRGB color values on every vertex to keep vertex colors".to_owned(),
            ),
        });
    }
}

fn unique_material_id(name: &str, ids_by_name: &mut BTreeMap<String, Id>) -> Result<Id> {
    if let Some(id) = ids_by_name.get(name) {
        return Ok(id.clone());
    }
    let mut base = name
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() || matches!(character, '_' | '-') {
                character.to_ascii_lowercase()
            } else {
                '_'
            }
        })
        .collect::<String>();
    if !base.as_bytes().first().is_some_and(u8::is_ascii_lowercase) {
        base.insert_str(0, "material_");
    }
    base.truncate(64);
    let mut candidate = base.clone();
    let mut suffix_number = 1_usize;
    while ids_by_name
        .values()
        .any(|id| id.as_str() == candidate.as_str())
    {
        suffix_number += 1;
        let suffix = format!("_{suffix_number}");
        candidate = format!("{}{suffix}", &base[..base.len().min(64 - suffix.len())]);
    }
    let id = Id::new(candidate)?;
    ids_by_name.insert(name.to_owned(), id.clone());
    Ok(id)
}

/// Sets the slots for each distinct raw `usemtl` name, retaining the external name separately.
fn finish_materials(
    mesh: &mut ExchangeMesh,
    face_materials: &[Option<usize>],
    material_names: &[String],
    ids_by_name: &mut BTreeMap<String, Id>,
    external_names: &mut BTreeMap<Id, String>,
) -> Result<()> {
    let mut keys = Vec::<Option<usize>>::new();
    let mut slots = Vec::<Option<Id>>::new();
    let mut indices = Vec::with_capacity(face_materials.len());
    for &material in face_materials {
        let key = material;
        let slot = if let Some(slot) = keys.iter().position(|&known| known == key) {
            slot
        } else {
            let id = match key.and_then(|index| material_names.get(index)) {
                None => None,
                Some(name) => {
                    let id = unique_material_id(name, ids_by_name)?;
                    external_names.insert(id.clone(), name.clone());
                    Some(id)
                }
            };
            keys.push(key);
            slots.push(id);
            slots.len() - 1
        };
        indices.push(u32::try_from(slot).unwrap_or(u32::MAX));
    }
    if slots.iter().all(Option::is_none) {
        return Ok(());
    }
    mesh.material_slots = slots;
    mesh.face_material_indices = indices;
    Ok(())
}

fn parse_floats<'a>(
    mut words: impl Iterator<Item = &'a str>,
    count: usize,
    line: usize,
) -> Result<Vec<f64>> {
    let mut result = Vec::with_capacity(count);
    for _ in 0..count {
        let value = words
            .next()
            .ok_or_else(|| parse_error(line))?
            .parse::<f64>()
            .map_err(|_| parse_error(line))?;
        if !value.is_finite() {
            return Err(parse_error(line));
        }
        result.push(value);
    }
    Ok(result)
}

/// Parses a `v` line: `x y z`, optionally with a `w` before the color, and an optional sRGB
/// `r g b` after the position. The color is decoded to linear light with alpha 1.
fn parse_vertex<'a>(
    words: impl Iterator<Item = &'a str>,
    line: usize,
) -> Result<([f64; 3], Option<[f64; 4]>)> {
    let values = words
        .map(|word| {
            word.parse::<f64>()
                .ok()
                .filter(|value| value.is_finite())
                .ok_or_else(|| parse_error(line))
        })
        .collect::<Result<Vec<_>>>()?;
    if !matches!(values.len(), 3 | 4 | 6 | 7) {
        return Err(parse_error(line));
    }
    let position = [values[0], values[1], values[2]];
    let color_start = match values.len() {
        6 => Some(3),
        7 => Some(4),
        _ => None,
    };
    let color = color_start.map(|start| {
        [
            srgb_decode(values[start]),
            srgb_decode(values[start + 1]),
            srgb_decode(values[start + 2]),
            1.0,
        ]
    });
    Ok((position, color))
}

fn parse_error(line: usize) -> PotError {
    PotError::with_details(
        ErrorCode::ImportFailed,
        "invalid OBJ syntax",
        json!({ "line": line + 1 }),
    )
}

fn obj_name(value: &str) -> String {
    value
        .chars()
        .map(|character| {
            if character.is_whitespace() {
                '_'
            } else {
                character
            }
        })
        .collect()
}

/// One `map_*` entry of an MTL material (Blender's `MTLTexMap`): the image as written, the MTL
/// directory it resolves against, and its `-o`, `-s`, and `-type` options.
#[derive(Clone, Debug)]
struct MtlMap {
    path: String,
    dir: PathBuf,
    translation: [f64; 3],
    scale: [f64; 3],
    sphere: bool,
}

impl MtlMap {
    fn new(dir: &Path) -> Self {
        Self {
            path: String::new(),
            dir: dir.to_path_buf(),
            translation: [0.0; 3],
            scale: [1.0; 3],
            sphere: false,
        }
    }
}

/// The texture slots an MTL material names, in Blender's `MTLTexMapType` order.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum MapKind {
    Color,
    Metallic,
    Specular,
    SpecularExponent,
    Roughness,
    Sheen,
    Reflection,
    Emission,
    Alpha,
    Normal,
}

/// One MTL material as read (Blender's `MTLMaterial`). A value stays negative until the file sets
/// it; Blender's importer then uses its default.
#[derive(Clone, Debug)]
struct MtlMaterial {
    spec_exponent: f64,
    ambient: [f64; 3],
    color: [f64; 3],
    spec: [f64; 3],
    emission: [f64; 3],
    ior: f64,
    alpha: f64,
    transmit: [f64; 3],
    roughness: f64,
    metallic: f64,
    sheen: f64,
    cc_thickness: f64,
    cc_roughness: f64,
    aniso: f64,
    aniso_rot: f64,
    illum: i32,
    normal_strength: f64,
    maps: BTreeMap<MapKind, MtlMap>,
}

impl Default for MtlMaterial {
    fn default() -> Self {
        Self {
            spec_exponent: -1.0,
            ambient: [-1.0; 3],
            color: [-1.0; 3],
            spec: [-1.0; 3],
            emission: [-1.0; 3],
            ior: -1.0,
            alpha: -1.0,
            transmit: [-1.0; 3],
            roughness: -1.0,
            metallic: -1.0,
            sheen: -1.0,
            cc_thickness: -1.0,
            cc_roughness: -1.0,
            aniso: -1.0,
            aniso_rot: -1.0,
            illum: -1,
            normal_strength: -1.0,
            maps: BTreeMap::new(),
        }
    }
}

/// Reads one MTL text into `library`, as Blender's `MTLParser::parse_and_store`: a `newmtl` opens
/// or reopens a material, and each later key sets its value. Keys before any `newmtl` are ignored.
fn parse_mtl(text: &str, dir: &Path, library: &mut BTreeMap<String, MtlMaterial>) {
    let mut current: Option<String> = None;
    for line in text.lines() {
        let (key, rest) = split_word(line);
        if key.is_empty() {
            continue;
        }
        if key == "newmtl" {
            let name = rest.trim().to_owned();
            library.entry(name.clone()).or_default();
            current = Some(name);
            continue;
        }
        let Some(name) = current.as_deref() else {
            continue;
        };
        let Some(material) = library.get_mut(name) else {
            continue;
        };
        match key {
            "Ns" => material.spec_exponent = number(rest, 324.0),
            "Ka" => material.ambient = take_numbers::<3>(rest, 0.0).0,
            "Kd" => material.color = take_numbers::<3>(rest, 0.8).0,
            "Ks" => material.spec = take_numbers::<3>(rest, 0.5).0,
            "Ke" => material.emission = take_numbers::<3>(rest, 0.0).0,
            "Ni" => material.ior = number(rest, 1.45),
            "d" => material.alpha = number(rest, 1.0),
            // Some files write `illum` as a float (#60135); Blender truncates it.
            "illum" => material.illum = number(rest, 1.0) as i32,
            "Pr" => material.roughness = number(rest, 0.5),
            "Pm" => material.metallic = number(rest, 0.0),
            "Ps" => material.sheen = number(rest, 0.0),
            "Pc" => material.cc_thickness = number(rest, 0.0),
            "Pcr" => material.cc_roughness = number(rest, 0.0),
            "aniso" => material.aniso = number(rest, 0.0),
            "anisor" => material.aniso_rot = number(rest, 0.0),
            "Kt" | "Tf" => material.transmit = take_numbers::<3>(rest, 0.0).0,
            _ => {
                if key.starts_with("map_") || key.starts_with("refl") || key.starts_with("bump") {
                    parse_map(key, rest, dir, material);
                }
            }
        }
    }
}

/// Reads a texture line: the map kind from its key, then its options, then the image path, which
/// is the rest of the line (`parse_texture_map` and `parse_texture_option`, `obj_import_file_reader.cc`).
fn parse_map(key: &str, text: &str, dir: &Path, material: &mut MtlMaterial) {
    let kind = match key {
        "map_Kd" => MapKind::Color,
        "map_Ks" => MapKind::Specular,
        "map_Ns" => MapKind::SpecularExponent,
        "map_d" => MapKind::Alpha,
        "refl" | "map_refl" => MapKind::Reflection,
        "map_Ke" => MapKind::Emission,
        "bump" | "map_Bump" | "map_bump" => MapKind::Normal,
        "map_Pr" => MapKind::Roughness,
        "map_Pm" => MapKind::Metallic,
        "map_Ps" => MapKind::Sheen,
        _ => return,
    };
    let mut map = material
        .maps
        .get(&kind)
        .cloned()
        .unwrap_or_else(|| MtlMap::new(dir));
    map.dir = dir.to_path_buf();
    let mut rest = text;
    loop {
        let (word, after) = split_word(rest);
        match word {
            "-o" => {
                let (values, next) = take_numbers::<3>(after, 0.0);
                map.translation = values;
                rest = next;
            }
            "-s" => {
                let (values, next) = take_numbers::<3>(after, 1.0);
                map.scale = values;
                rest = next;
            }
            "-bm" => {
                let (values, next) = take_numbers::<1>(after, 1.0);
                material.normal_strength = values[0];
                rest = next;
            }
            "-type" => {
                map.sphere = true;
                rest = skip_words(after, 1);
            }
            "-blendu" | "-blendv" | "-boost" | "-cc" | "-clamp" | "-imfchan" | "-texres" => {
                rest = skip_words(after, 1);
            }
            "-mm" => rest = skip_words(after, 2),
            "-t" => rest = skip_words(after, 3),
            _ => break,
        }
    }
    map.path = rest.trim().replace('"', "");
    material.maps.insert(kind, map);
}

/// The first whitespace-separated word of `text` and the text after it.
fn split_word(text: &str) -> (&str, &str) {
    let text = text.trim_start();
    match text.find(char::is_whitespace) {
        Some(end) => (&text[..end], &text[end..]),
        None => (text, ""),
    }
}

/// The text after `count` words.
fn skip_words(mut text: &str, count: usize) -> &str {
    for _ in 0..count {
        text = split_word(text).1;
    }
    text
}

fn parse_number(word: &str) -> Option<f64> {
    word.parse::<f64>().ok().filter(|value| value.is_finite())
}

/// The first number of `text`, or `default` when it is missing or not a number.
fn number(text: &str, default: f64) -> f64 {
    parse_number(split_word(text).0).unwrap_or(default)
}

/// Up to `N` leading numbers of `text`, with `default` for any that is missing, and the text after
/// the numbers read.
fn take_numbers<const N: usize>(text: &str, default: f64) -> ([f64; N], &str) {
    let mut values = [default; N];
    let mut rest = text;
    for value in &mut values {
        let (word, after) = split_word(rest);
        let Some(number) = parse_number(word) else {
            break;
        };
        *value = number;
        rest = after;
    }
    (values, rest)
}

/// The MTL map of `kind` when it names a file.
fn map_of(mtl: &MtlMaterial, kind: MapKind) -> Option<&MtlMap> {
    mtl.maps.get(&kind).filter(|map| !map.path.is_empty())
}

/// The flags Blender's importer reads from `illum` (`set_bsdf_socket_values`, `obj_import_mtl.cc`).
#[derive(Default)]
struct IllumFlags {
    highlight: bool,
    transparency: bool,
    reflection: bool,
    glass: bool,
}

fn illum_flags(illum: i32) -> IllumFlags {
    match illum {
        2 => IllumFlags {
            highlight: true,
            ..IllumFlags::default()
        },
        3 | 5 | 8 => IllumFlags {
            reflection: true,
            ..IllumFlags::default()
        },
        4 => IllumFlags {
            reflection: true,
            transparency: true,
            glass: true,
            ..IllumFlags::default()
        },
        6 | 7 => IllumFlags {
            reflection: true,
            transparency: true,
            ..IllumFlags::default()
        },
        9 => IllumFlags {
            transparency: true,
            glass: true,
            ..IllumFlags::default()
        },
        _ => IllumFlags::default(),
    }
}

fn import_loss(
    losses: &mut Vec<Loss>,
    id: &Id,
    feature: &str,
    reason: impl Into<String>,
    suggestion: Option<&str>,
) {
    losses.push(Loss {
        feature_id: feature.to_owned(),
        data_id: Some(id.as_str().to_owned()),
        reason: reason.into(),
        suggestion: suggestion.map(str::to_owned),
    });
}

/// The Potter material for one MTL entry, from Blender's `set_bsdf_socket_values`
/// (`obj_import_mtl.cc`). A linked texture replaces its socket's value, so the matching factor is
/// set to one. Values Potter's material fields cannot hold are losses.
fn potter_material(
    mtl: &MtlMaterial,
    id: &Id,
    doc: &mut SceneDoc,
    images: &mut BTreeMap<PathBuf, Id>,
    losses: &mut Vec<Loss>,
) -> Result<Material> {
    let flags = illum_flags(mtl.illum);
    let mut material = Material {
        name: id.as_str().to_owned(),
        ..Material::default()
    };

    let mut specular = mean3(mtl.spec);
    if specular < 0.0 {
        specular = if flags.highlight { 1.0 } else { 0.0 };
    }
    #[expect(
        clippy::float_cmp,
        reason = "exact sentinel: Blender's default specular level of 0.5 imports without a loss"
    )]
    if specular != 0.5 {
        import_loss(
            losses,
            id,
            "material.specular",
            "MTL specular level has no Potter material field; it is not imported",
            None,
        );
    }

    let mut roughness = if mtl.spec_exponent < 0.0 {
        if flags.highlight { 0.0 } else { 1.0 }
    } else {
        1.0 - (mtl.spec_exponent.clamp(0.0, 1000.0) / 1000.0).sqrt()
    };
    let mut metallic = if flags.reflection {
        let ambient = mean3(mtl.ambient);
        if ambient < 0.0 { 1.0 } else { ambient }
    } else {
        0.0
    };
    if mtl.roughness >= 0.0 {
        roughness = mtl.roughness;
    }
    if mtl.metallic >= 0.0 {
        metallic = mtl.metallic;
    }
    material.roughness = roughness.clamp(0.0, 1.0);
    material.metallic = metallic.clamp(0.0, 1.0);

    let transmission = mean3(mtl.transmit);
    if transmission >= 0.0 {
        material.transmission = transmission.clamp(0.0, 1.0);
    }

    let mut ior = mtl.ior;
    if ior < 0.0 {
        if flags.transparency {
            ior = 1.0;
        }
        if flags.glass {
            ior = 1.5;
        }
    }
    if ior > 0.0 {
        material.ior = ior;
    }

    if mtl.color.iter().all(|channel| *channel >= 0.0) {
        material.base_color = [
            mtl.color[0],
            mtl.color[1],
            mtl.color[2],
            material.base_color[3],
        ];
    }
    let mut alpha = mtl.alpha;
    if flags.transparency && alpha < 0.0 {
        alpha = 1.0;
    }
    #[expect(
        clippy::float_cmp,
        reason = "exact sentinel: -1.0 is the unset alpha of MtlMaterial's default"
    )]
    if alpha != -1.0 {
        material.base_color[3] = alpha.clamp(0.0, 1.0);
    }
    if flags.transparency || (0.0..1.0).contains(&alpha) {
        "blend".clone_into(&mut material.alpha_mode);
    }

    let color_path = resolve_map(map_of(mtl, MapKind::Color), id, losses);
    let alpha_path = resolve_map(map_of(mtl, MapKind::Alpha), id, losses);
    let color = texture_ref(doc, images, color_path.clone(), id, losses)?;
    if let Some(reference) = color.clone() {
        material.base_color_texture = Some(reference);
        material.base_color = [1.0, 1.0, 1.0, material.base_color[3]];
    }
    if alpha_path.is_some() {
        // Blender sets BLEND for an alpha map. Potter reads alpha from the base colour texture, so
        // only the same image carries it.
        "blend".clone_into(&mut material.alpha_mode);
        if color.is_some() && alpha_path == color_path {
            material.base_color[3] = 1.0;
        } else {
            import_loss(
                losses,
                id,
                "material.alpha_texture",
                "MTL alpha texture is not the base colour texture; Potter reads alpha from the base colour texture, so only d is imported",
                Some("use one image for map_Kd and map_d"),
            );
        }
    }

    // Reflection is linked to Metallic after Metallic (Blender's `MTLTexMapType` order).
    let metallic_path = resolve_map(
        map_of(mtl, MapKind::Reflection).or_else(|| map_of(mtl, MapKind::Metallic)),
        id,
        losses,
    );
    if let Some(reference) = texture_ref(doc, images, metallic_path, id, losses)? {
        material.metallic_texture = Some(reference);
        material.metallic = 1.0;
    }
    // Roughness is linked after SpecularExponent, so `map_Pr` wins over `map_Ns`.
    let roughness_path = resolve_map(
        map_of(mtl, MapKind::Roughness).or_else(|| map_of(mtl, MapKind::SpecularExponent)),
        id,
        losses,
    );
    if let Some(reference) = texture_ref(doc, images, roughness_path, id, losses)? {
        material.roughness_texture = Some(reference);
        material.roughness = 1.0;
    }

    if map_of(mtl, MapKind::Specular).is_some() {
        import_loss(
            losses,
            id,
            "material.specular_texture",
            "MTL specular map has no Potter material field; it is not imported",
            None,
        );
    }

    let emission_map = map_of(mtl, MapKind::Emission).is_some();
    // Blender's `set_bsdf_socket_values`: colours above 1 are split into a unit colour and a
    // strength. A map sets the strength to one.
    let (emission_color, emission_strength) = if mtl.emission.iter().all(|channel| *channel >= 0.0)
    {
        let peak = mtl.emission[0].max(mtl.emission[1]).max(mtl.emission[2]);
        if peak > 1.0 {
            (mtl.emission.map(|channel| channel / peak), peak)
        } else {
            (mtl.emission, 1.0)
        }
    } else {
        ([0.0; 3], if emission_map { 1.0 } else { 0.0 })
    };
    material.emission_color = emission_color;
    material.emission_strength = emission_strength;
    if emission_map {
        import_loss(
            losses,
            id,
            "material.emission_texture",
            "MTL emission map has no Potter material field; it is not imported",
            None,
        );
    }

    // Blender's Normal Map Strength is max(0, -bm), and no -bm leaves zero: a normal map with zero
    // strength has no effect, so it is not imported.
    let strength = mtl.normal_strength.max(0.0);
    if strength > 0.0 {
        let normal_path = resolve_map(map_of(mtl, MapKind::Normal), id, losses);
        if let Some(reference) = texture_ref(doc, images, normal_path, id, losses)? {
            material.normal_texture = Some(reference);
            #[expect(
                clippy::float_cmp,
                reason = "exact check: Potter's normal scale is exactly 1, and any other strength is reported"
            )]
            if strength != 1.0 {
                import_loss(
                    losses,
                    id,
                    "material.normal_strength",
                    format!(
                        "MTL bump strength {strength} is not Potter's normal scale of 1; it is imported as 1"
                    ),
                    None,
                );
            }
        }
    }

    if map_of(mtl, MapKind::Sheen).is_some() || mtl.sheen >= 0.0 {
        import_loss(
            losses,
            id,
            "material.sheen",
            "MTL sheen has no Potter material field; it is not imported",
            None,
        );
    }
    if mtl.cc_thickness >= 0.0 || mtl.cc_roughness >= 0.0 {
        import_loss(
            losses,
            id,
            "material.coat",
            "MTL clearcoat has no Potter material field; it is not imported",
            None,
        );
    }
    if mtl.aniso >= 0.0 || mtl.aniso_rot >= 0.0 {
        import_loss(
            losses,
            id,
            "material.anisotropy",
            "MTL anisotropy has no Potter material field; it is not imported",
            None,
        );
    }
    #[expect(
        clippy::float_cmp,
        reason = "exact default check: only a zero offset and unit scale are dropped, so no tolerance applies"
    )]
    if mtl
        .maps
        .values()
        .any(|map| map.translation != [0.0; 3] || map.scale != [1.0; 3])
    {
        import_loss(
            losses,
            id,
            "material.texture_transform",
            "MTL texture -o/-s options have no Potter texture field; the transform is not imported",
            None,
        );
    }
    if mtl.maps.values().any(|map| map.sphere) {
        import_loss(
            losses,
            id,
            "material.texture_projection",
            "MTL sphere projection has no Potter texture field; the map is imported as flat",
            None,
        );
    }
    Ok(material)
}

/// The file a map names, found as Blender's `load_texture_image` finds it: beside the MTL as
/// written, then as written, then with underscores as spaces, then by its basename beside the MTL.
/// `None` when the map names no file or none of those exists (a loss).
fn resolve_map(map: Option<&MtlMap>, id: &Id, losses: &mut Vec<Loss>) -> Option<PathBuf> {
    let map = map?;
    let written = map.path.as_str();
    let basename = Path::new(written)
        .file_name()
        .map(PathBuf::from)
        .unwrap_or_default();
    let candidates = [
        map.dir.join(written),
        PathBuf::from(written),
        map.dir.join(written.replace('_', " ")),
        map.dir.join(basename),
    ];
    if let Some(found) = candidates.into_iter().find(|candidate| candidate.is_file()) {
        Some(fs::canonicalize(&found).unwrap_or(found))
    } else {
        losses.push(Loss {
            feature_id: "material.texture_missing".to_owned(),
            data_id: Some(id.as_str().to_owned()),
            reason: format!("MTL texture `{written}` was not found; the map is not imported"),
            suggestion: Some("place the image beside the .mtl file".to_owned()),
        });
        None
    }
}

/// The texture reference for a resolved map file. Each file is loaded once, as an sRGB image like
/// Blender's default colour space for loaded images (`load_image_at_path`).
fn texture_ref(
    doc: &mut SceneDoc,
    images: &mut BTreeMap<PathBuf, Id>,
    path: Option<PathBuf>,
    id: &Id,
    losses: &mut Vec<Loss>,
) -> Result<Option<TextureRef>> {
    let Some(path) = path else {
        return Ok(None);
    };
    if let Some(image) = images.get(&path) {
        return Ok(Some(reference(image.clone())));
    }
    let bytes = match fs::read(&path) {
        Ok(bytes) => bytes,
        Err(error) => {
            import_loss(
                losses,
                id,
                "material.texture_missing",
                format!("MTL texture `{}` cannot be read: {error}", path.display()),
                None,
            );
            return Ok(None);
        }
    };
    let (width, height, _) = match crate::image::decode_pixels(&bytes, ImageColorspace::Srgb) {
        Ok(decoded) => decoded,
        Err(error) => {
            import_loss(
                losses,
                id,
                "material.texture_codec",
                format!(
                    "MTL texture `{}` cannot be decoded: {}",
                    path.display(),
                    error.message
                ),
                None,
            );
            return Ok(None);
        }
    };
    let name = path
        .file_stem()
        .map(|stem| stem.to_string_lossy().into_owned())
        .unwrap_or_default();
    let image_id = unique_image_id(doc, &name)?;
    doc.images.insert(
        image_id.clone(),
        Image {
            name,
            source: ImageSource::File,
            colorspace: ImageColorspace::Srgb,
            width,
            height,
            tiles: Vec::new(),
            tiled: false,
            blob: None,
            source_path: Some(path.to_string_lossy().into_owned()),
            source_hash: Some(crate::hash::sha256(&bytes)),
            alpha_mode: ImageAlphaMode::Straight,
        },
    );
    images.insert(path, image_id.clone());
    Ok(Some(reference(image_id)))
}

fn reference(image: Id) -> TextureRef {
    TextureRef {
        image,
        uv_layer: None,
        interpolation: ImageInterpolation::Linear,
        extension: ImageExtension::Repeat,
    }
}

/// A Potter image ID from a file stem: lowercase, `[a-z0-9_-]`, starting with a letter, and unused.
fn unique_image_id(doc: &SceneDoc, name: &str) -> Result<Id> {
    let mut base = name
        .chars()
        .map(|character| {
            let character = character.to_ascii_lowercase();
            if character.is_ascii_lowercase()
                || character.is_ascii_digit()
                || character == '_'
                || character == '-'
            {
                character
            } else {
                '_'
            }
        })
        .take(48)
        .collect::<String>();
    if !base.starts_with(|character: char| character.is_ascii_lowercase()) {
        base.insert_str(0, "image_");
    }
    let mut candidate = base.clone();
    let mut suffix = 2_u32;
    while doc.images.contains_key(&Id::new(candidate.clone())?) {
        candidate = format!("{base}_{suffix}");
        suffix = suffix
            .checked_add(1)
            .ok_or_else(|| PotError::new(ErrorCode::LimitExceeded, "image ID counter overflow"))?;
    }
    Id::new(candidate)
}

/// Binds each imported object's material slots to its data block: the slot list, and each face's
/// slot index. Objects follow the root collection in import order.
fn bind_slots(doc: &mut SceneDoc, bindings: Vec<(Vec<Option<Id>>, Vec<u32>)>) -> Result<()> {
    let root = Id::new("collection_root")?;
    let objects = doc
        .collections
        .get(&root)
        .ok_or_else(|| PotError::new(ErrorCode::InternalError, "root collection is missing"))?
        .objects
        .clone();
    for (node_id, (slots, indices)) in objects.iter().zip(bindings) {
        let data_id = doc
            .nodes
            .get(node_id)
            .and_then(|node| node.data.clone())
            .ok_or_else(|| {
                PotError::new(
                    ErrorCode::InternalError,
                    "imported object has no data block",
                )
            })?;
        let data = doc.data_blocks.get_mut(&data_id).ok_or_else(|| {
            PotError::new(ErrorCode::InternalError, "imported data block is missing")
        })?;
        data.materials = slots;
        if let Some(mesh) = data.mesh.as_mut() {
            for (face, index) in mesh.faces.iter_mut().zip(indices) {
                face.material_index = index;
            }
        }
    }
    Ok(())
}

pub(crate) fn imported_graph(path: &Path, scene_id: String) -> Result<ImportedGraph> {
    let ObjImport {
        meshes,
        losses,
        materials: library,
        external_material_names,
    } = import(path)?;
    let bindings = meshes
        .iter()
        .map(|mesh| {
            (
                mesh.material_slots.clone(),
                mesh.face_material_indices.clone(),
            )
        })
        .collect::<Vec<_>>();
    let slot_ids = meshes
        .iter()
        .flat_map(|mesh| mesh.material_slots.iter().flatten().cloned())
        .collect::<BTreeSet<Id>>();
    let mut graph = import_graph_from_meshes(meshes, path, scene_id, "obj", losses)?;
    let mut images = BTreeMap::<PathBuf, Id>::new();
    for id in &slot_ids {
        let external_name = external_material_names
            .get(id)
            .map_or(id.as_str(), String::as_str);
        // A usemtl name with no MTL entry gets the default material, as Blender creates one.
        let mtl = library.get(external_name).cloned().unwrap_or_default();
        let mut material =
            potter_material(&mtl, id, &mut graph.doc, &mut images, &mut graph.losses)?;
        external_name.clone_into(&mut material.name);
        graph.doc.materials.insert(id.clone(), material);
    }
    bind_slots(&mut graph.doc, bindings)?;
    graph.doc.validate()?;
    Ok(graph)
}

#[cfg(test)]
mod tests {
    use super::{
        MtlMaterial, export, illum_flags, illum_mode, import, imported_graph, material_diagnostics,
        parse_mtl, potter_material, srgb_encode,
    };
    use crate::{
        error::{ErrorCode, PotError, Result},
        exchange::test_support::triangle_with_layers,
        geom::color_layers::{ColorDomain, ColorLayer, ColorStore},
        image::{ImageExtension, ImageInterpolation},
        model::{
            Id, Image, ImageAlphaMode, ImageColorspace, ImageSource, Material, SceneDoc, TextureRef,
        },
    };
    use std::{collections::BTreeMap, fs, path::Path};

    const SCENE_ID: &str = "00000000-0000-4000-8000-000000000000";

    fn empty_doc() -> SceneDoc {
        SceneDoc::new(SCENE_ID.to_owned())
    }

    fn io<T>(result: std::io::Result<T>) -> Result<T> {
        result.map_err(|error| PotError::io(&error))
    }

    /// A triangle bound to one material slot, and the document holding that material.
    fn bound_triangle(material: Material) -> Result<(SceneDoc, crate::exchange::ExchangeMesh)> {
        let id = Id::new("shiny")?;
        let mut doc = empty_doc();
        doc.materials.insert(id.clone(), material);
        let mut mesh = triangle_with_layers();
        mesh.material_slots = vec![Some(id)];
        mesh.face_material_indices = vec![0; mesh.faces.len()];
        Ok((doc, mesh))
    }

    fn mtl_block(text: &str, name: &str) -> Vec<String> {
        text.lines()
            .skip_while(|line| *line != format!("newmtl {name}"))
            .map(str::to_owned)
            .collect()
    }

    #[test]
    fn averages_corner_colors_when_writing_obj_vertices() -> Result<()> {
        // Blender's POINT lookup averages CORNER data (`BKE_attribute.hh:644-668`; `mesh_attributes.cc:24-44`).
        let mut mesh = triangle_with_layers();
        mesh.faces.push(vec![0, 1, 2]);
        mesh.smooth.push(false);
        mesh.face_material_indices.push(0);
        mesh.normals = Some(vec![[0.0, 0.0, 1.0]; 6]);
        mesh.uv_active = None;
        mesh.colors = ColorStore {
            layers: vec![ColorLayer {
                name: "Corner".to_owned(),
                domain: ColorDomain::Corner,
                values: vec![[1.0, 0.0, 0.0, 1.0]; 3]
                    .into_iter()
                    .chain(vec![[0.0, 0.0, 1.0, 1.0]; 3])
                    .collect(),
            }],
            active: Some("Corner".to_owned()),
            render: Some("Corner".to_owned()),
        };
        let exported = export(&empty_doc(), Path::new(""), &[mesh])?;
        let text = String::from_utf8_lossy(&exported.object);
        let first_vertex = text
            .lines()
            .find(|line| line.starts_with("v "))
            .ok_or_else(|| PotError::new(ErrorCode::InternalError, "OBJ vertex is missing"))?;
        let color = first_vertex
            .split_whitespace()
            .skip(4)
            .take(3)
            .map(|value| {
                value
                    .parse::<f64>()
                    .map_err(|error| PotError::new(ErrorCode::InternalError, error.to_string()))
            })
            .collect::<Result<Vec<_>>>()?;
        assert!((color[0] - srgb_encode(0.5)).abs() < 1.0e-6);
        assert_eq!(color[1], 0.0);
        assert!((color[2] - srgb_encode(0.5)).abs() < 1.0e-6);
        Ok(())
    }

    #[test]
    fn negative_obj_vertex_colors_are_clamped_and_reported() -> Result<()> {
        // Blender rejects color layers with a negative vertex RGB (`obj_import_file_reader.cc:92-105`; `obj_import_mesh.cc:463-474`).
        let mut mesh = triangle_with_layers();
        mesh.colors = ColorStore {
            layers: vec![ColorLayer {
                name: "Color".to_owned(),
                domain: ColorDomain::Point,
                values: vec![[-0.25, 0.25, 1.0, 1.0]; 3],
            }],
            active: Some("Color".to_owned()),
            render: Some("Color".to_owned()),
        };
        let exported = export(&empty_doc(), Path::new(""), std::slice::from_ref(&mesh))?;
        let object = String::from_utf8_lossy(&exported.object);
        for line in object.lines().filter(|line| line.starts_with("v ")) {
            let color = line.split_whitespace().skip(4).collect::<Vec<_>>();
            assert_eq!(color[0], "0");
            assert!(
                color
                    .iter()
                    .all(|channel| { channel.parse::<f64>().is_ok_and(|value| value >= 0.0) })
            );
        }
        let losses = crate::exchange::element_data_losses("obj", std::slice::from_ref(&mesh));
        assert!(losses.iter().any(|loss| {
            loss.feature_id == "mesh.color_attribute" && loss.reason.contains("negative")
        }));
        Ok(())
    }

    #[test]
    fn imports_the_entire_obj_g_remainder_as_one_group_name() -> Result<()> {
        // Blender treats the `g` line remainder as one group name (`obj_import_file_reader.cc:398-408,632-649`).
        let file = tempfile::NamedTempFile::new().map_err(|error| PotError::io(&error))?;
        io(fs::write(
            file.path(),
            "o object\nv 0 0 0\nv 1 0 0\nv 0 1 0\ng Front Group\nf 1 2 3\n",
        ))?;
        let imported = import(file.path())?;
        let group = imported.meshes[0]
            .vertex_groups
            .first()
            .ok_or_else(|| PotError::new(ErrorCode::InternalError, "OBJ group is missing"))?;
        assert_eq!(group.name, "Front Group");
        assert_eq!(group.weights, vec![(0, 1.0), (1, 1.0), (2, 1.0)]);
        Ok(())
    }

    #[test]
    fn exports_one_group_per_face_and_reports_overlapping_memberships() -> Result<()> {
        // Blender writes one deform group per face (`obj_export_file_writer.cc:402-413`).
        let mut mesh = triangle_with_layers();
        mesh.vertex_groups = ["Group Name", "Second"]
            .into_iter()
            .map(|name| crate::exchange::ExchangeVertexGroup {
                name: name.to_owned(),
                weights: vec![(0, 1.0), (1, 1.0), (2, 1.0)],
            })
            .collect();
        let meshes = [mesh];
        let exported = export(&empty_doc(), Path::new(""), &meshes)?;
        let object = String::from_utf8_lossy(&exported.object);
        assert!(object.lines().any(|line| line == "g Group Name"));
        assert!(!object.lines().any(|line| line == "g Group Name Second"));
        let losses = crate::exchange::element_data_losses("obj", &meshes);
        assert!(losses.iter().any(|loss| {
            loss.feature_id == "mesh.vertex_group" && loss.reason.contains("Second")
        }));
        Ok(())
    }

    #[test]
    fn imports_blender_material_names_and_resolves_mtl_by_raw_name() -> Result<()> {
        // Blender preserves the raw `usemtl` name for material lookup (`obj_import_file_reader.cc:657-664`).
        let dir = io(tempfile::tempdir())?;
        let object_path = dir.path().join("scene.obj");
        io(fs::write(
            &object_path,
            "mtllib scene.mtl\no object\nv 0 0 0\nv 1 0 0\nv 0 1 0\nusemtl Material.001\nf 1 2 3\nusemtl material_001\nf 1 3 2\n",
        ))?;
        io(fs::write(
            dir.path().join("scene.mtl"),
            "newmtl Material.001\nKd 1 0 0\nnewmtl material_001\nKd 0 1 0\n",
        ))?;
        let imported = import(&object_path)?;
        let slots = &imported.meshes[0].material_slots;
        assert_eq!(slots.len(), 2);
        assert_ne!(slots[0], slots[1]);
        assert_eq!(slots[0].as_ref().map(Id::as_str), Some("material_001"));
        assert_eq!(slots[1].as_ref().map(Id::as_str), Some("material_001_2"));

        let graph = imported_graph(&object_path, SCENE_ID.to_owned())?;
        let (red_id, red) = graph
            .doc
            .materials
            .iter()
            .find(|(_, material)| material.name == "Material.001")
            .ok_or_else(|| PotError::new(ErrorCode::InternalError, "raw OBJ material missing"))?;
        let (green_id, green) = graph
            .doc
            .materials
            .iter()
            .find(|(_, material)| material.name == "material_001")
            .ok_or_else(|| {
                PotError::new(ErrorCode::InternalError, "second OBJ material missing")
            })?;
        assert_eq!(red_id.as_str(), "material_001");
        assert_eq!(green_id.as_str(), "material_001_2");
        assert_eq!(red.base_color[..3], [1.0, 0.0, 0.0]);
        assert_eq!(green.base_color[..3], [0.0, 1.0, 0.0]);
        let data = graph
            .doc
            .data_blocks
            .values()
            .find(|data| data.materials.len() == 2)
            .ok_or_else(|| {
                PotError::new(ErrorCode::InternalError, "bound material slots missing")
            })?;
        let slots = &data.materials;
        assert_eq!(
            graph.doc.materials[slots[0].as_ref().ok_or_else(|| {
                PotError::new(ErrorCode::InternalError, "first material slot is empty")
            })?]
            .name,
            "Material.001"
        );
        assert_eq!(
            graph.doc.materials[slots[1].as_ref().ok_or_else(|| {
                PotError::new(ErrorCode::InternalError, "second material slot is empty")
            })?]
            .name,
            "material_001"
        );
        let imported_mesh = data
            .mesh
            .as_ref()
            .ok_or_else(|| PotError::new(ErrorCode::InternalError, "imported mesh is missing"))?;
        assert_eq!(
            imported_mesh
                .faces
                .iter()
                .map(|face| face.material_index)
                .collect::<Vec<_>>(),
            vec![0, 1]
        );
        Ok(())
    }

    #[test]
    fn unassigned_obj_faces_use_empty_usemtl() -> Result<()> {
        // Blender clears the material with an empty name (`obj_export_file_writer.cc:56-62`).
        let exported = export(&empty_doc(), Path::new(""), &[triangle_with_layers()])?;
        let object = String::from_utf8_lossy(&exported.object);
        assert!(object.lines().any(|line| line == "usemtl"));
        assert_eq!(exported.material.len(), 0);
        Ok(())
    }

    #[test]
    fn valid_default_like_material_ids_are_written_verbatim() -> Result<()> {
        let mut doc = empty_doc();
        let first = Id::new("potter_default")?;
        let second = Id::new("potter_default_material")?;
        doc.materials.insert(first.clone(), Material::default());
        doc.materials.insert(second.clone(), Material::default());
        let mut mesh = triangle_with_layers();
        mesh.faces.push(mesh.faces[0].clone());
        mesh.smooth.push(false);
        mesh.normals = Some(vec![[0.0, 0.0, 1.0]; 6]);
        mesh.uv_active = None;
        mesh.material_slots = vec![Some(first), Some(second)];
        mesh.face_material_indices = vec![0, 1];
        let exported = export(&doc, Path::new(""), &[mesh])?;
        let object = String::from_utf8_lossy(&exported.object);
        let mtl = String::from_utf8_lossy(&exported.material);
        assert!(object.lines().any(|line| line == "usemtl potter_default"));
        assert!(
            object
                .lines()
                .any(|line| line == "usemtl potter_default_material")
        );
        assert!(mtl.lines().any(|line| line == "newmtl potter_default"));
        assert!(
            mtl.lines()
                .any(|line| line == "newmtl potter_default_material")
        );
        Ok(())
    }

    #[test]
    fn map_d_packs_source_opacity_into_png_alpha() -> Result<()> {
        // Blender connects the image Alpha output to Principled Alpha (`obj_import_mtl.cc:424-436`).
        let dir = io(tempfile::tempdir())?;
        let image_path = dir.path().join("opacity.pixels");
        let pixels = crate::image::encode_pixels(1, 1, &[[0.2, 0.4, 0.6, 0.25]])?;
        io(fs::write(&image_path, &pixels))?;
        let image_id = Id::new("opacity")?;
        let mut doc = empty_doc();
        doc.images.insert(
            image_id.clone(),
            Image {
                name: "Opacity".to_owned(),
                source: ImageSource::File,
                colorspace: ImageColorspace::Srgb,
                width: 1,
                height: 1,
                tiles: Vec::new(),
                tiled: false,
                blob: None,
                source_path: Some(image_path.to_string_lossy().into_owned()),
                source_hash: Some(crate::hash::sha256(&pixels)),
                alpha_mode: ImageAlphaMode::default(),
            },
        );
        let material_id = Id::new("shiny")?;
        doc.materials.insert(
            material_id.clone(),
            Material {
                name: "shiny".to_owned(),
                alpha_mode: "blend".to_owned(),
                base_color_texture: Some(TextureRef {
                    image: image_id,
                    uv_layer: None,
                    interpolation: ImageInterpolation::Linear,
                    extension: ImageExtension::Repeat,
                }),
                ..Material::default()
            },
        );
        let mut mesh = triangle_with_layers();
        mesh.material_slots = vec![Some(material_id)];
        mesh.face_material_indices = vec![0];
        let exported = export(&doc, dir.path(), &[mesh])?;
        let mtl = String::from_utf8_lossy(&exported.material);
        let texture_name = mtl
            .lines()
            .find(|line| line.starts_with("map_d "))
            .and_then(|line| line.split_whitespace().last())
            .ok_or_else(|| PotError::new(ErrorCode::InternalError, "MTL map_d is missing"))?;
        let png = exported
            .textures
            .iter()
            .find(|(name, _)| name.as_str() == texture_name)
            .map(|(_, bytes)| bytes)
            .ok_or_else(|| PotError::new(ErrorCode::InternalError, "map_d PNG is missing"))?;
        let (_, _, decoded) = crate::image::decode_pixels(png, ImageColorspace::Linear)?;
        assert!((decoded[0][3] - 0.25).abs() < 0.005);
        Ok(())
    }

    #[test]
    fn material_diagnostics_report_losses_without_loading_texture_pixels() -> Result<()> {
        let dir = io(tempfile::tempdir())?;
        let mut doc = empty_doc();
        let image_id = Id::new("missing_image")?;
        doc.images.insert(
            image_id.clone(),
            Image {
                name: "Missing".to_owned(),
                source: ImageSource::File,
                colorspace: ImageColorspace::Srgb,
                width: 1,
                height: 1,
                tiles: Vec::new(),
                tiled: false,
                blob: None,
                source_path: Some(
                    dir.path()
                        .join("missing.pixels")
                        .to_string_lossy()
                        .into_owned(),
                ),
                source_hash: None,
                alpha_mode: ImageAlphaMode::default(),
            },
        );
        let material_id = Id::new("shiny")?;
        doc.materials.insert(
            material_id.clone(),
            Material {
                name: "shiny".to_owned(),
                transmission: 0.5,
                base_color_texture: Some(TextureRef {
                    image: image_id,
                    uv_layer: None,
                    interpolation: ImageInterpolation::Linear,
                    extension: ImageExtension::Repeat,
                }),
                ..Material::default()
            },
        );
        let mut mesh = triangle_with_layers();
        mesh.material_slots = vec![Some(material_id)];
        mesh.face_material_indices = vec![0];
        let losses = material_diagnostics(&doc, dir.path(), &[mesh])?;
        assert!(
            losses
                .iter()
                .any(|loss| loss.feature_id == "material.mtl.transmission")
        );
        Ok(())
    }

    #[test]
    fn writes_the_active_layer_as_blender_v_with_corner_indices() -> Result<()> {
        let exported = export(&empty_doc(), Path::new(""), &[triangle_with_layers()])?;
        let text = String::from_utf8_lossy(&exported.object);
        let vt = text
            .lines()
            .filter(|line| line.starts_with("vt "))
            .collect::<Vec<_>>();
        // Lightmap is active; its storage V flips to Blender's V.
        assert_eq!(
            vt,
            [
                "vt 0.200000 0.900000",
                "vt 0.600000 0.700000",
                "vt 0.400000 1.000000"
            ]
        );
        let face = text
            .lines()
            .find(|line| line.starts_with("f "))
            .ok_or_else(|| PotError::new(ErrorCode::SceneInvalid, "OBJ face is missing"))?;
        let corners = face.split_whitespace().skip(1).collect::<Vec<_>>();
        assert_eq!(corners.len(), 3);
        for (index, corner) in corners.iter().enumerate() {
            let fields = corner.split('/').collect::<Vec<_>>();
            assert_eq!(fields.len(), 3);
            assert_eq!(fields[1], (index + 1).to_string());
            assert_ne!(fields[0], "");
            assert_ne!(fields[2], "");
        }
        Ok(())
    }

    #[test]
    fn imports_vt_as_the_uv_layer_in_storage_convention() -> Result<()> {
        let original = triangle_with_layers();
        let exported = export(&empty_doc(), Path::new(""), std::slice::from_ref(&original))?;
        let file = tempfile::NamedTempFile::new().map_err(|error| PotError::io(&error))?;
        fs::write(file.path(), exported.object).map_err(|error| PotError::io(&error))?;
        let imported = import(file.path())?;
        let layer = imported.meshes[0]
            .uv_layers
            .first()
            .ok_or_else(|| PotError::new(ErrorCode::InternalError, "OBJ import has no UV layer"))?;
        assert_eq!(layer.name, "UVMap");
        // The export writes the active layer, Lightmap, so that is the layer that comes back.
        for (read, written) in layer.corners[0]
            .iter()
            .zip(&original.uv_layers[1].corners[0])
        {
            assert!((read[0] - written[0]).abs() < 1.0e-6);
            assert!((read[1] - written[1]).abs() < 1.0e-6);
        }
        Ok(())
    }

    #[test]
    fn empty_declared_object_keeps_uv_metadata_when_texture_coordinates_exist() -> Result<()> {
        let file = tempfile::NamedTempFile::new().map_err(|error| PotError::io(&error))?;
        fs::write(file.path(), "o empty\nvt 0.25\n").map_err(|error| PotError::io(&error))?;
        let imported = import(file.path())?;
        let mesh = imported
            .meshes
            .first()
            .ok_or_else(|| PotError::new(ErrorCode::InternalError, "OBJ object is missing"))?;
        assert_eq!(mesh.faces, Vec::<Vec<usize>>::new());
        assert_eq!(mesh.uv_layers.len(), 1);
        assert_eq!(mesh.uv_layers[0].name, "UVMap");
        assert_eq!(mesh.uv_layers[0].corners, Vec::<Vec<[f64; 2]>>::new());
        assert_eq!(mesh.uv_active, Some(0));
        assert_eq!(mesh.uv_render, Some(0));
        Ok(())
    }

    #[test]
    fn vertices_without_faces_import_as_one_object_named_by_the_last_o() -> Result<()> {
        let file = tempfile::NamedTempFile::new().map_err(|error| PotError::io(&error))?;
        fs::write(file.path(), "o first\nv 0 0 0\no cloud\nv 1 0 0\nv 0 1 0\n")
            .map_err(|error| PotError::io(&error))?;
        let imported = import(file.path())?;
        assert_eq!(imported.meshes.len(), 1);
        let mesh = &imported.meshes[0];
        assert_eq!(mesh.name, "cloud");
        assert_eq!(mesh.positions.len(), 3);
        assert_eq!(mesh.faces.len(), 0);
        Ok(())
    }

    #[test]
    fn writes_blender_principled_values_as_mtl_lines() -> Result<()> {
        let (doc, mesh) = bound_triangle(Material {
            name: "shiny".to_owned(),
            base_color: [0.1, 0.2, 0.3, 1.0],
            roughness: 0.2,
            ..Material::default()
        })?;
        let exported = export(&doc, Path::new(""), &[mesh])?;
        let mtl = String::from_utf8(exported.material)
            .map_err(|error| PotError::new(ErrorCode::InternalError, error.to_string()))?;
        // Ns = (1 - roughness)^2 * 1000 with roughness 0.2; Ka = 1 without metal; illum 2 for a
        // specular dielectric that is opaque.
        assert_eq!(
            mtl_block(&mtl, "shiny"),
            [
                "newmtl shiny",
                "Ns 640.000000",
                "Ka 1.000000 1.000000 1.000000",
                "Kd 0.100000 0.200000 0.300000",
                "Ks 0.500000 0.500000 0.500000",
                "Ke 0.000000 0.000000 0.000000",
                "Ni 1.500000",
                "d 1.000000",
                "illum 2",
            ]
        );
        Ok(())
    }

    #[test]
    fn metal_transparency_selects_illum_six_and_writes_emission_and_alpha() -> Result<()> {
        let (doc, mesh) = bound_triangle(Material {
            name: "shiny".to_owned(),
            base_color: [0.1, 0.2, 0.3, 0.5],
            alpha_mode: "blend".to_owned(),
            metallic: 1.0,
            roughness: 0.0,
            emission_color: [2.0, 1.0, 0.0],
            emission_strength: 0.5,
            ..Material::default()
        })?;
        let exported = export(&doc, Path::new(""), &[mesh])?;
        let mtl = String::from_utf8(exported.material)
            .map_err(|error| PotError::new(ErrorCode::InternalError, error.to_string()))?;
        // Ke is emission colour times strength, unnormalised; Ka is the metallic value.
        assert_eq!(
            mtl_block(&mtl, "shiny"),
            [
                "newmtl shiny",
                "Ns 1000.000000",
                "Ka 1.000000 1.000000 1.000000",
                "Kd 0.100000 0.200000 0.300000",
                "Ks 0.500000 0.500000 0.500000",
                "Ke 1.000000 0.500000 0.000000",
                "Ni 1.500000",
                "d 0.500000",
                "illum 6",
            ]
        );
        Ok(())
    }

    #[test]
    fn illum_follows_blender_export_and_import_rules() {
        // Export: 1 without specular, 3 or 6 for metal (6 when transparent), 9 transparent, else 2.
        assert_eq!(illum_mode(0.0, 0.0, false), 1);
        assert_eq!(illum_mode(0.5, 0.0, false), 2);
        assert_eq!(illum_mode(0.5, 0.0, true), 9);
        assert_eq!(illum_mode(0.5, 1.0, false), 3);
        assert_eq!(illum_mode(0.5, 1.0, true), 6);
        // Import: illum 6 is reflection plus refraction, 9 is glass without reflection.
        let six = illum_flags(6);
        assert!(six.reflection && six.transparency && !six.glass);
        let nine = illum_flags(9);
        assert!(nine.glass && nine.transparency && !nine.reflection);
        assert!(illum_flags(2).highlight);
        assert!(!illum_flags(1).reflection);
    }

    #[test]
    fn imports_mtl_values_by_blender_formulas() -> Result<()> {
        let mut library = BTreeMap::<String, MtlMaterial>::new();
        parse_mtl(
            "newmtl shiny\nNs 500\nKa 0.25 0.25 0.25\nKd 0.1 0.2 0.3\nKe 4 2 1\nNi 1.33\nd 0.5\nillum 3\n",
            Path::new(""),
            &mut library,
        );
        let id = Id::new("shiny")?;
        let mut doc = empty_doc();
        let mut images = BTreeMap::new();
        let mut losses = Vec::new();
        let material = potter_material(&library["shiny"], &id, &mut doc, &mut images, &mut losses)?;
        // roughness = 1 - sqrt(Ns / 1000); illum 3 is reflection, so metallic is the Ka average.
        assert!((material.roughness - (1.0 - 0.5_f64.sqrt())).abs() < 1.0e-9);
        assert!((material.metallic - 0.25).abs() < 1.0e-9);
        assert_eq!(material.base_color, [0.1, 0.2, 0.3, 0.5]);
        // d below one is transparent, which Blender writes as BLEND.
        assert_eq!(material.alpha_mode, "blend");
        assert!((material.ior - 1.33).abs() < 1.0e-9);
        // Ke above one is split into a unit colour and a strength of its peak.
        assert_eq!(material.emission_color, [1.0, 0.5, 0.25]);
        assert!((material.emission_strength - 4.0).abs() < 1.0e-9);
        // Ks is absent, so Blender's specular is zero here, which Potter cannot hold.
        assert!(
            losses
                .iter()
                .any(|loss| loss.feature_id == "material.specular")
        );
        Ok(())
    }

    #[test]
    fn shared_map_kd_and_map_d_image_becomes_the_base_colour_texture() -> Result<()> {
        let dir = io(tempfile::tempdir())?;
        let pixels = crate::image::encode_pixels(1, 1, &[[1.0, 0.0, 0.0, 0.5]])?;
        io(fs::write(dir.path().join("tex.pixels"), pixels))?;
        let mut library = BTreeMap::<String, MtlMaterial>::new();
        parse_mtl(
            "newmtl shiny\nmap_Kd tex.pixels\nmap_d tex.pixels\n",
            dir.path(),
            &mut library,
        );
        let id = Id::new("shiny")?;
        let mut doc = empty_doc();
        let mut images = BTreeMap::new();
        let mut losses = Vec::new();
        let material = potter_material(&library["shiny"], &id, &mut doc, &mut images, &mut losses)?;
        let reference = material
            .base_color_texture
            .ok_or_else(|| PotError::new(ErrorCode::InternalError, "base colour map missing"))?;
        assert!(doc.images.contains_key(&reference.image));
        assert_eq!(material.base_color, [1.0, 1.0, 1.0, 1.0]);
        assert_eq!(material.alpha_mode, "blend");
        assert!(
            !losses
                .iter()
                .any(|loss| loss.feature_id == "material.alpha_texture")
        );
        Ok(())
    }

    #[test]
    fn exported_mtl_imports_back_with_the_same_principled_values() -> Result<()> {
        let (doc, mesh) = bound_triangle(Material {
            name: "shiny".to_owned(),
            base_color: [0.1, 0.2, 0.3, 1.0],
            roughness: 0.2,
            metallic: 0.5,
            ..Material::default()
        })?;
        let exported = export(&doc, Path::new(""), &[mesh])?;
        let dir = io(tempfile::tempdir())?;
        io(fs::write(dir.path().join("scene.obj"), &exported.object))?;
        io(fs::write(dir.path().join("scene.mtl"), &exported.material))?;
        let graph = imported_graph(&dir.path().join("scene.obj"), SCENE_ID.to_owned())?;
        let material =
            graph.doc.materials.get(&Id::new("shiny")?).ok_or_else(|| {
                PotError::new(ErrorCode::InternalError, "imported material missing")
            })?;
        // Ns is written with six decimals, so the roughness comes back within 1e-5.
        assert!((material.roughness - 0.2).abs() < 1.0e-5);
        assert!((material.metallic - 0.5).abs() < 1.0e-6);
        for (read, written) in material.base_color[..3].iter().zip([0.1, 0.2, 0.3]) {
            assert!((read - written).abs() < 1.0e-6);
        }
        assert!((material.ior - 1.5).abs() < 1.0e-6);
        assert_eq!(material.alpha_mode, "opaque");
        Ok(())
    }
}
