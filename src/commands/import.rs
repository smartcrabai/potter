use std::{collections::BTreeMap, fs, path::Path};

use serde_json::{Map, Value, json};

use crate::{
    cli::{AssetPolicy, ImportArgs, ImportMode},
    commands::util::{check_base_revision, check_extension, exchange_format_name, scene_changes},
    error::{ErrorCode, PotError, Result},
    exchange::{self, ImportedGraph, Loss},
    hash,
    model::{Id, SceneDoc},
    response::SceneInfo,
    store::Project,
};

pub fn run(args: &ImportArgs) -> Result<(Option<SceneInfo>, Value)> {
    let format = exchange_format_name(args.format);
    check_extension(&args.file, format, "input extension does not match format")?;
    if !matches!(
        format,
        "blend"
            | "glb"
            | "gltf"
            | "usda"
            | "usdc"
            | "usd"
            | "usdz"
            | "alembic"
            | "fbx"
            | "fbx-binary"
            | "bvh"
            | "obj"
            | "ply"
            | "stl"
            | "svg"
    ) {
        return Err(PotError::with_details(
            ErrorCode::UnsupportedFeature,
            format!("{format} import is not implemented"),
            json!({ "feature_id": format!("format.{format}"), "status": "not_supported" }),
        ));
    }
    if !args.file.is_file() {
        return Err(PotError::new(
            ErrorCode::FileNotFound,
            "import source file does not exist",
        ));
    }
    let mut project = Project::open_exclusive(&args.scene)?;
    check_base_revision(project.doc(), args.base_revision)?;
    let imported = load_graph(format, &args.file, args.blender.as_deref())?;
    if !imported.losses.is_empty() && !args.allow_lossy {
        return Err(lossy_error(format, &imported.losses));
    }
    let packed_resource_hashes = imported
        .doc
        .resources
        .values()
        .filter(|resource| resource["source_packed"] == true)
        .filter_map(|resource| resource["hash"].as_str().map(str::to_owned))
        .collect::<std::collections::BTreeSet<_>>();
    let mut asset_blobs = BTreeMap::new();
    for (digest, bytes) in imported.assets {
        if format == "blend"
            && args.asset_policy == AssetPolicy::Link
            && !packed_resource_hashes.contains(&digest)
        {
            continue;
        }
        if hash::sha256(&bytes) != digest {
            return Err(PotError::with_details(
                ErrorCode::ImportFailed,
                "imported asset content does not match its hash",
                json!({"hash":digest}),
            ));
        }
        if asset_blobs
            .get(&digest)
            .is_some_and(|existing| existing != &bytes)
        {
            return Err(PotError::with_details(
                ErrorCode::ImportFailed,
                "conflicting bytes were returned for one imported asset hash",
                json!({"hash":digest}),
            ));
        }
        asset_blobs.entry(digest).or_insert(bytes);
    }
    let before = project.doc().clone();
    let mut candidate = imported.doc.clone();
    let id_mappings = match args.mode {
        ImportMode::Replace => {
            candidate.scene_id.clone_from(&before.scene_id);
            candidate.revision = before.revision;
            candidate.history = before.history.clone();
            imported.id_mappings.clone()
        }
        ImportMode::Append => {
            let append_mappings = append_graph(&mut candidate, &before, &args.file)?;
            if imported
                .id_mappings
                .as_object()
                .is_some_and(|values| !values.is_empty())
            {
                json!({ "source": imported.id_mappings, "append": append_mappings })
            } else {
                append_mappings
            }
        }
    };
    candidate.scene_id.clone_from(&before.scene_id);
    candidate.revision = before.revision;
    candidate.history = before.history.clone();
    let (external_resources, resource_warnings) = if format == "blend" {
        configure_blender_resources(&mut candidate, &asset_blobs, args.asset_policy)
    } else {
        (Vec::new(), Vec::new())
    };
    let source_bytes = fs::read(&args.file).map_err(|error| PotError::io(&error))?;
    let source_name = args
        .file
        .file_name()
        .and_then(|value| value.to_str())
        .unwrap_or("source");
    let source_hash = hash::sha256(&source_bytes);
    let mut resources = vec![blob_record(source_name, &source_hash)];
    let mut compatibility = resources.clone();
    for (name, bytes) in &imported.compat_blobs {
        let digest = hash::sha256(bytes);
        let record = blob_record(name, &digest);
        resources.push(record.clone());
        compatibility.push(record);
    }
    for digest in asset_blobs.keys() {
        resources.push(asset_record(digest));
    }
    resources.extend(external_resources);
    let metadata_key = format!(
        "exchange_import_r{}_{}",
        before.revision,
        safe_namespace(source_name)
    );
    candidate.compatibility.insert(
        metadata_key,
        json!({
            "source": imported.source,
            "blobs": compatibility,
            "format": format,
        }),
    );
    let changes = scene_changes(&before, &candidate)?;
    candidate.validate()?;
    let candidate_revision = before
        .revision
        .checked_add(1)
        .ok_or_else(|| PotError::new(ErrorCode::LimitExceeded, "revision limit reached"))?;
    let operations = json!({
        "kind": "import",
        "format": format,
        "file": args.file,
        "mode": if args.mode == ImportMode::Append { "append" } else { "replace" },
        "asset_policy": if args.asset_policy == AssetPolicy::Copy { "copy" } else { "link" },
    });
    let result = json!({
        "format": format,
        "source": imported.source,
        "committed": !args.dry_run,
        "base_revision": before.revision,
        "candidate_revision": candidate_revision,
        "changes": changes,
        "id_mappings": id_mappings,
        "resources": resources,
        "compatibility": compatibility,
        "migrations": [],
        "losses": imported.losses,
        "warnings": resource_warnings,
    });
    if args.dry_run {
        return Ok((Some(project.info()?), result));
    }
    project.write_compat_blob(source_name, &source_bytes)?;
    for (name, bytes) in &imported.compat_blobs {
        project.write_compat_blob(name, bytes)?;
    }
    let scene = project.commit_import_with_assets(candidate, operations, changes, &asset_blobs)?;
    Ok((Some(scene), result))
}

fn load_graph(format: &str, path: &Path, blender: Option<&Path>) -> Result<ImportedGraph> {
    let scene_id = uuid::Uuid::new_v4().to_string();
    match format {
        "blend" => exchange::blend::import_blend(path, blender),
        "glb" | "gltf" => exchange::gltf::import(path, scene_id),
        "usda" => exchange::usd::import_usda(path, scene_id),
        "usdc" | "usd" => exchange::usdc::import(path, scene_id, blender),
        "usdz" => exchange::usd::import_usdz(path, scene_id),
        "alembic" => exchange::alembic::import(path, scene_id, blender),
        "bvh" => exchange::bvh::import(path, scene_id),
        "fbx" => exchange::fbx::import(path, scene_id),
        "fbx-binary" => exchange::fbx_binary::import(path, scene_id),
        "obj" => exchange::obj::imported_graph(path, scene_id),
        "svg" => exchange::svg::import(path, scene_id),
        "stl" => exchange::stl::imported_graph(path, scene_id),
        "ply" => exchange::ply::imported_graph(path, scene_id),
        _ => Err(PotError::new(
            ErrorCode::UnsupportedFeature,
            format!("{format} import is not implemented"),
        )),
    }
}

fn lossy_error(format: &str, losses: &[Loss]) -> PotError {
    PotError::with_details(
        ErrorCode::UnrepresentableFeature,
        format!("{format} import cannot preserve all source features"),
        json!({ "result": { "format": format, "losses": losses } }),
    )
}

fn safe_namespace(name: &str) -> String {
    let mut result = String::with_capacity(name.len());
    for character in name
        .bytes()
        .map(char::from)
        .map(|value| value.to_ascii_lowercase())
    {
        if character.is_ascii_lowercase()
            || character.is_ascii_digit()
            || character == '_'
            || character == '-'
        {
            result.push(character);
        } else {
            result.push('_');
        }
    }
    if result.is_empty() || !result.as_bytes()[0].is_ascii_lowercase() {
        result.insert_str(0, "source_");
    }
    result.truncate(48);
    result
}

fn blob_record(name: &str, digest: &str) -> Value {
    let filename = Path::new(name)
        .file_name()
        .and_then(|value| value.to_str())
        .unwrap_or("blob");
    let hex = digest.strip_prefix("sha256:").unwrap_or_default();
    json!({ "name": filename, "hash": digest, "path": format!("compat/sha256/{hex}/{filename}") })
}
fn asset_record(digest: &str) -> Value {
    let hex = digest.strip_prefix("sha256:").unwrap_or_default();
    json!({ "name": "blob", "hash": digest, "path": format!("assets/sha256/{hex}/blob") })
}

fn configure_blender_resources(
    doc: &mut SceneDoc,
    staged_assets: &BTreeMap<String, Vec<u8>>,
    policy: AssetPolicy,
) -> (Vec<Value>, Vec<Value>) {
    let sources = doc
        .resources
        .iter()
        .map(|(id, value)| (id.clone(), value.clone()))
        .collect::<Vec<_>>();
    let mut reference_uris = BTreeMap::new();
    let mut result = Vec::with_capacity(sources.len());
    let mut warnings = Vec::new();
    for (id, source) in sources {
        let original_path = source
            .get("original_path")
            .and_then(Value::as_str)
            .map(str::to_owned);
        let digest = source
            .get("hash")
            .and_then(Value::as_str)
            .map(str::to_owned);
        let filename = source
            .get("filename")
            .and_then(Value::as_str)
            .filter(|name| {
                Path::new(name).file_name().and_then(|part| part.to_str()) == Some(*name)
            })
            .unwrap_or("resource");
        let source_packed = source["source_packed"] == true;
        let copy = policy == AssetPolicy::Copy || source_packed;
        let available = source["status"] != "missing"
            && digest
                .as_deref()
                .is_some_and(|digest| staged_assets.contains_key(digest));
        let uri = if copy && available {
            let digest = digest.as_deref().unwrap_or_default();
            let hex = digest.strip_prefix("sha256:").unwrap_or_default();
            format!("assets/sha256/{hex}/{filename}")
        } else {
            original_path
                .clone()
                .unwrap_or_else(|| source["uri"].as_str().unwrap_or_default().to_owned())
        };
        let missing = source["status"] == "missing" || (copy && !available);
        let status = if missing { "missing" } else { "available" };
        if missing {
            warnings.push(json!({
                "code": "ASSET_MISSING",
                "resource_id": id,
                "uri": uri,
                "message": "referenced Blender asset was not found; import preserved the missing link"
            }));
        }
        if let Some(original_path) = original_path.as_ref() {
            reference_uris.insert(original_path.clone(), uri.clone());
        }
        if let Some(resource) = doc.resources.get_mut(&id) {
            resource["uri"] = json!(uri);
            resource["status"] = json!(status);
            resource["packed"] = json!(false);
        }
        result.push(json!({
            "id": id,
            "uri": uri,
            "original_path": original_path,
            "hash": digest,
            "kind": source["kind"],
            "owner": source["owner"],
            "status": status,
            "packed": false,
        }));
    }
    for data in doc.data_blocks.values_mut() {
        let Some(text) = data.text.as_mut() else {
            continue;
        };
        if let Some(uri) = reference_uris.get(&text.font) {
            text.font.clone_from(uri);
        }
    }
    for scene in doc.scenes.values_mut() {
        for strip in &mut scene.sequencer.strips {
            if let Some(source) = strip.source.as_ref()
                && let Some(uri) = reference_uris.get(source)
            {
                strip.source.clone_from(&Some(uri.clone()));
            }
        }
    }
    (result, warnings)
}

fn append_graph(imported: &mut SceneDoc, base: &SceneDoc, source_path: &Path) -> Result<Value> {
    let namespace = safe_namespace(
        source_path
            .file_stem()
            .and_then(|value| value.to_str())
            .unwrap_or("import"),
    );
    let target_root = base
        .scenes
        .get(&base.active_scene)
        .ok_or_else(|| PotError::new(ErrorCode::SceneInvalid, "active scene is missing"))?
        .root_collection
        .clone();
    let imported_root = imported
        .scenes
        .get(&imported.active_scene)
        .ok_or_else(|| PotError::new(ErrorCode::SceneInvalid, "imported active scene is missing"))?
        .root_collection
        .clone();

    let node_map = registry_mapping(imported.nodes.keys(), base.nodes.keys(), &namespace)?;
    let data_map = registry_mapping(
        imported.data_blocks.keys(),
        base.data_blocks.keys(),
        &namespace,
    )?;
    let material_map =
        registry_mapping(imported.materials.keys(), base.materials.keys(), &namespace)?;
    let action_map = registry_mapping(imported.actions.keys(), base.actions.keys(), &namespace)?;
    let image_map = registry_mapping(imported.images.keys(), base.images.keys(), &namespace)?;
    let world_map = registry_mapping(imported.worlds.keys(), base.worlds.keys(), &namespace)?;
    let node_group_map = registry_mapping(
        imported.node_groups.keys(),
        base.node_groups.keys(),
        &namespace,
    )?;
    let resource_map =
        registry_mapping(imported.resources.keys(), base.resources.keys(), &namespace)?;
    let library_map =
        registry_mapping(imported.libraries.keys(), base.libraries.keys(), &namespace)?;
    let mut collection_map = registry_mapping(
        imported.collections.keys(),
        base.collections.keys(),
        &namespace,
    )?;
    collection_map.insert(imported_root.clone(), target_root.clone());
    let root_collection = imported.collections.remove(&imported_root).ok_or_else(|| {
        PotError::new(
            ErrorCode::SceneInvalid,
            "imported root collection is missing",
        )
    })?;

    let incoming_collections = std::mem::take(&mut imported.collections);
    let mut merged_collections = base.collections.clone();
    for (old_id, mut collection) in incoming_collections {
        let new_id = collection_map.get(&old_id).cloned().ok_or_else(|| {
            PotError::new(ErrorCode::InternalError, "collection ID mapping is missing")
        })?;
        remap_vec(&mut collection.children, &collection_map);
        remap_vec(&mut collection.objects, &node_map);
        merged_collections.insert(new_id, collection);
    }
    let mut merged_root = merged_collections
        .get(&target_root)
        .cloned()
        .ok_or_else(|| {
            PotError::new(ErrorCode::SceneInvalid, "target root collection is missing")
        })?;
    merged_root.objects.extend(
        root_collection
            .objects
            .iter()
            .filter_map(|id| node_map.get(id).cloned()),
    );
    merged_root.children.extend(
        root_collection
            .children
            .iter()
            .filter_map(|id| collection_map.get(id).cloned()),
    );
    merged_collections.insert(target_root.clone(), merged_root);

    let incoming_nodes = std::mem::take(&mut imported.nodes);
    let mut merged_nodes = base.nodes.clone();
    for (old_id, mut node) in incoming_nodes {
        let new_id = node_map
            .get(&old_id)
            .cloned()
            .ok_or_else(|| PotError::new(ErrorCode::InternalError, "node ID mapping is missing"))?;
        node.data = remap_option(node.data, &data_map);
        node.parent = remap_option(node.parent, &node_map);
        node.materials = remap_vec_owned(node.materials, &material_map);
        node.action = remap_option(node.action, &action_map);
        for track in &mut node.nla_tracks {
            for strip in &mut track.strips {
                strip.action = action_map
                    .get(&strip.action)
                    .cloned()
                    .unwrap_or_else(|| strip.action.clone());
            }
        }
        for constraint in &mut node.constraints {
            constraint.target = remap_option(constraint.target.clone(), &node_map);
        }
        for driver in &mut node.drivers {
            for variable in &mut driver.variables {
                variable.target = node_map
                    .get(&variable.target)
                    .cloned()
                    .unwrap_or_else(|| variable.target.clone());
            }
        }
        merged_nodes.insert(new_id, node);
    }
    let mut incoming_data = std::mem::take(&mut imported.data_blocks);
    for data in incoming_data.values_mut() {
        if let Some(grease_pencil) = &mut data.grease_pencil {
            for layer in &mut grease_pencil.layers {
                for frame in &mut layer.frames {
                    for stroke in &mut frame.strokes {
                        stroke.material = remap_option(stroke.material.clone(), &material_map);
                    }
                }
            }
        }
        if let Some(shape_keys) = &mut data.shape_keys {
            shape_keys.action = remap_option(shape_keys.action.clone(), &action_map);
        }
        if let Some(volume) = &mut data.volume
            && let crate::geom::volume::VolumeSource::File(source) = &mut volume.source
        {
            let remapped = source
                .content_ref
                .as_ref()
                .and_then(|reference| Id::new(reference.clone()).ok())
                .and_then(|old_id| {
                    resource_map
                        .get(&old_id)
                        .map(|new_id| new_id.as_str().to_owned())
                });
            if let Some(resource_id) = remapped {
                source.content_ref = Some(resource_id);
            }
        }
    }
    let merged_data = merge_registry(incoming_data, &base.data_blocks, &data_map)?;
    let merged_worlds = merge_registry(
        std::mem::take(&mut imported.worlds),
        &base.worlds,
        &world_map,
    )?;
    let merged_images = merge_registry(
        std::mem::take(&mut imported.images),
        &base.images,
        &image_map,
    )?;
    let mut incoming_materials = std::mem::take(&mut imported.materials);
    for material in incoming_materials.values_mut() {
        for texture in [
            &mut material.base_color_texture,
            &mut material.roughness_texture,
            &mut material.metallic_texture,
            &mut material.normal_texture,
        ]
        .into_iter()
        .flatten()
        {
            texture.image = image_map.get(&texture.image).cloned().ok_or_else(|| {
                PotError::with_details(
                    ErrorCode::ImportFailed,
                    "imported material references a missing image",
                    json!({"image":texture.image}),
                )
            })?;
        }
    }
    let merged_materials = merge_registry(incoming_materials, &base.materials, &material_map)?;
    let mut incoming_actions = std::mem::take(&mut imported.actions);
    for action in incoming_actions.values_mut() {
        for slot in &mut action.slots {
            if let Some(mapped) = node_map.get(&slot.node) {
                slot.node.clone_from(mapped);
            }
        }
    }
    let merged_actions = merge_registry(incoming_actions, &base.actions, &action_map)?;
    let merged_node_groups = merge_registry(
        std::mem::take(&mut imported.node_groups),
        &base.node_groups,
        &node_group_map,
    )?;
    let merged_resources = merge_registry(
        std::mem::take(&mut imported.resources),
        &base.resources,
        &resource_map,
    )?;
    let merged_libraries = merge_registry(
        std::mem::take(&mut imported.libraries),
        &base.libraries,
        &library_map,
    )?;
    let mut compatibility = base.compatibility.clone();
    for (key, value) in std::mem::take(&mut imported.compatibility) {
        compatibility.insert(format!("{namespace}_{key}"), value);
    }
    imported.active_scene = base.active_scene.clone();
    imported.scenes = base.scenes.clone();
    imported.collections = merged_collections;
    imported.nodes = merged_nodes;
    imported.data_blocks = merged_data;
    imported.images = merged_images;
    imported.materials = merged_materials;
    imported.actions = merged_actions;
    imported.node_groups = merged_node_groups;
    imported.worlds = merged_worlds;
    imported.resources = merged_resources;
    imported.libraries = merged_libraries;
    imported.compatibility = compatibility;

    let mut mappings = Map::new();
    mappings.insert("worlds".to_owned(), mapping_value(&world_map));
    mappings.insert("collections".to_owned(), mapping_value(&collection_map));
    mappings.insert("nodes".to_owned(), mapping_value(&node_map));
    mappings.insert("data_blocks".to_owned(), mapping_value(&data_map));
    mappings.insert("materials".to_owned(), mapping_value(&material_map));
    mappings.insert("images".to_owned(), mapping_value(&image_map));
    mappings.insert("actions".to_owned(), mapping_value(&action_map));
    mappings.insert("node_groups".to_owned(), mapping_value(&node_group_map));
    mappings.insert("resources".to_owned(), mapping_value(&resource_map));
    mappings.insert("libraries".to_owned(), mapping_value(&library_map));
    Ok(Value::Object(mappings))
}

fn registry_mapping<'a>(
    incoming: impl Iterator<Item = &'a Id>,
    existing: impl Iterator<Item = &'a Id>,
    namespace: &str,
) -> Result<BTreeMap<Id, Id>> {
    let mut used = existing.cloned().collect::<std::collections::BTreeSet<_>>();
    let mut mappings = BTreeMap::new();
    for id in incoming {
        let mut candidate = id.clone();
        let mut suffix = 0_usize;
        while used.contains(&candidate) {
            candidate = namespaced_id(namespace, id, suffix)?;
            suffix = suffix.saturating_add(1);
        }
        used.insert(candidate.clone());
        mappings.insert(id.clone(), candidate);
    }
    Ok(mappings)
}

fn namespaced_id(namespace: &str, id: &Id, suffix: usize) -> Result<Id> {
    let suffix = if suffix == 0 {
        String::new()
    } else {
        format!("_{suffix}")
    };
    let available = 63_usize
        .saturating_sub(namespace.len())
        .saturating_sub(suffix.len());
    let id_part = id.as_str().chars().take(available).collect::<String>();
    Id::new(format!("{namespace}_{id_part}{suffix}"))
}

fn merge_registry<T: Clone>(
    mut incoming: BTreeMap<Id, T>,
    existing: &BTreeMap<Id, T>,
    mapping: &BTreeMap<Id, Id>,
) -> Result<BTreeMap<Id, T>> {
    let mut merged = existing.clone();
    for (old_id, value) in std::mem::take(&mut incoming) {
        let new_id = mapping.get(&old_id).cloned().ok_or_else(|| {
            PotError::new(ErrorCode::InternalError, "registry ID mapping is missing")
        })?;
        merged.insert(new_id, value);
    }
    Ok(merged)
}

fn remap_option(value: Option<Id>, mapping: &BTreeMap<Id, Id>) -> Option<Id> {
    value.map(|id| mapping.get(&id).cloned().unwrap_or(id))
}

fn remap_vec(values: &mut Vec<Id>, mapping: &BTreeMap<Id, Id>) {
    for value in values {
        if let Some(mapped) = mapping.get(value) {
            value.clone_from(mapped);
        }
    }
}

fn remap_vec_owned(values: Vec<Id>, mapping: &BTreeMap<Id, Id>) -> Vec<Id> {
    values
        .into_iter()
        .map(|id| mapping.get(&id).cloned().unwrap_or(id))
        .collect()
}

fn mapping_value(mapping: &BTreeMap<Id, Id>) -> Value {
    Value::Object(
        mapping
            .iter()
            .filter(|(old, new)| *old != *new)
            .map(|(old, new)| (old.to_string(), json!(new)))
            .collect(),
    )
}
