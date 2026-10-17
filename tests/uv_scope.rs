//! UV layer scope and store hygiene through the batch boundary. `mesh.delete` and `mesh.separate`
//! leave no rows for faces they remove. `uv.layer_rename` and `uv.layer_delete` reach only the
//! items the edited mesh users reach. A layer named `ACTIVE` is not a modifier selector. A paint
//! without a selector samples the active layer, and an empty selector is rejected.

#![expect(clippy::unwrap_used, reason = "tests use fixed valid scenes")]

use std::path::Path;

use potter_core::{
    error::{ErrorCode, PotError},
    geom::Mesh,
    graph::{GraphKind, GraphNode, NodeGroup},
    image::ImageInterpolation,
    model::{Id, Modifier, ModifierRuntime, SceneDoc, TextureRef},
    ops::{ApplyOutcome, OP_NAMES, apply_batch, apply_batch_with_project_context},
    schema::schema,
};
use serde_json::{Value, json};
use tempfile::tempdir;

const SCENE_ID: &str = "6f1c2a4e-3b5d-4c8e-9a7f-1d2e3f4a5b6c";

fn empty_scene() -> SceneDoc {
    SceneDoc::new(SCENE_ID.to_owned())
}

fn batch(doc: &SceneDoc, operations: Value) -> Value {
    let mut batch = json!({"schema_version": 1, "base_revision": doc.revision});
    batch["operations"] = operations;
    batch
}

fn apply(doc: &SceneDoc, operations: Value) -> Result<SceneDoc, PotError> {
    Ok(apply_batch(doc, &batch(doc, operations))?.doc)
}

fn apply_at(doc: &SceneDoc, operations: Value, root: &Path) -> Result<ApplyOutcome, PotError> {
    apply_batch_with_project_context(doc, &batch(doc, operations), root, None)
}

fn node_mesh<'a>(doc: &'a SceneDoc, node: &str) -> &'a Mesh {
    let data = doc.nodes[&Id::new(node).unwrap()].data.as_ref().unwrap();
    doc.data_blocks[data].mesh.as_ref().unwrap()
}

fn node_store(doc: &SceneDoc, node: &str) -> Value {
    node_mesh(doc, node).attributes["uv_layers"].clone()
}

fn face_ids(doc: &SceneDoc, node: &str) -> Vec<u32> {
    node_mesh(doc, node)
        .faces
        .iter()
        .map(|face| face.id)
        .collect()
}

fn layer_names(doc: &SceneDoc, node: &str) -> Vec<String> {
    node_mesh(doc, node)
        .attributes
        .get("uv_layers")
        .and_then(|store| store.get("layers"))
        .and_then(Value::as_array)
        .map_or_else(Vec::new, |layers| {
            layers
                .iter()
                .map(|layer| layer["name"].as_str().unwrap().to_owned())
                .collect()
        })
}

fn modifier_param(doc: &SceneDoc, node: &str, key: &str) -> Value {
    doc.nodes[&Id::new(node).unwrap()].modifiers[0].params[key].clone()
}

fn texture_layer(doc: &SceneDoc, material: &str) -> Option<String> {
    doc.materials[&Id::new(material).unwrap()]
        .base_color_texture
        .as_ref()
        .unwrap()
        .uv_layer
        .clone()
}

fn set_store(doc: &mut SceneDoc, node: &str, store: Value) {
    let data = doc.nodes[&Id::new(node).unwrap()].data.clone().unwrap();
    let mesh = doc
        .data_blocks
        .get_mut(&data)
        .unwrap()
        .mesh
        .as_mut()
        .unwrap();
    mesh.attributes.insert("uv_layers".to_owned(), store);
}

fn add_modifier(doc: &mut SceneDoc, node: &str, modifier_type: &str, params: Value) {
    let Value::Object(params) = params else {
        unreachable!("modifier test parameters must be an object")
    };
    let node = doc.nodes.get_mut(&Id::new(node).unwrap()).unwrap();
    node.modifiers.push(Modifier {
        id: Id::new(format!("{modifier_type}_modifier")).unwrap(),
        modifier_type: modifier_type.to_owned(),
        name: modifier_type.to_owned(),
        enabled: true,
        show_viewport: true,
        show_render: true,
        params,
        binding_data: None,
        runtime: ModifierRuntime::default(),
    });
}

/// Creates material `material` whose base-color texture samples layer `layer`.
fn textured_material(doc: &mut SceneDoc, material: &str, layer: &str) {
    *doc = apply(doc, json!([{"op":"material.create","id":material}])).unwrap();
    doc.materials
        .get_mut(&Id::new(material).unwrap())
        .unwrap()
        .base_color_texture = Some(TextureRef {
        image: Id::new("paint").unwrap(),
        uv_layer: Some(layer.to_owned()),
        interpolation: ImageInterpolation::Linear,
    });
}

fn attach(doc: &mut SceneDoc, node: &str, material: &str) {
    doc.nodes
        .get_mut(&Id::new(node).unwrap())
        .unwrap()
        .materials
        .push(Some(Id::new(material).unwrap()));
}

/// Two unit boxes `a` and `b`, each with a `Light` layer and a `uv_project` modifier naming it.
fn two_lit_boxes() -> SceneDoc {
    let mut doc = apply(
        &empty_scene(),
        json!([
            {"op":"node.create","id":"a","kind":"box","params":{"size":1}},
            {"op":"node.create","id":"b","kind":"box","params":{"size":1}},
            {"op":"uv.layer_create","target":{"id":"a"},"name":"Light"},
            {"op":"uv.layer_create","target":{"id":"b"},"name":"Light"}
        ]),
    )
    .unwrap();
    add_modifier(&mut doc, "a", "uv_project", json!({"uv_layer":"Light"}));
    add_modifier(&mut doc, "b", "uv_project", json!({"uv_layer":"Light"}));
    doc
}

/// Corner `corner` of face `face` in the `Base` layer, in V-down storage.
fn base_uv(face: u32, corner: u32) -> [f64; 2] {
    [
        f64::from(face) * 0.125 + f64::from(corner) * 0.0625,
        0.25 + f64::from(corner) * 0.125,
    ]
}

/// Corner `corner` of face `face` in the `Detail` layer, in V-down storage.
fn detail_uv(face: u32, corner: u32) -> [f64; 2] {
    [
        0.5 + f64::from(face) * 0.0625,
        0.75 - f64::from(corner) * 0.125,
    ]
}

/// The two-layer store of a textured box over `face_ids`, with `Detail` active and `Base` render.
/// Face 2 is fully pinned in `Base`; face 3 has its first and last corners pinned in `Detail`.
fn textured_store(face_ids: &[u32]) -> Value {
    let base: Vec<Value> = face_ids
        .iter()
        .map(|&face| {
            let uv: Vec<[f64; 2]> = (0..4).map(|corner| base_uv(face, corner)).collect();
            let mut entry = json!({"face_id": face, "uv": uv});
            if face == 2 {
                entry["pinned"] = json!([true, true, true, true]);
            }
            entry
        })
        .collect();
    let detail: Vec<Value> = face_ids
        .iter()
        .map(|&face| {
            let uv: Vec<[f64; 2]> = (0..4).map(|corner| detail_uv(face, corner)).collect();
            let mut entry = json!({"face_id": face, "uv": uv});
            if face == 3 {
                entry["pinned"] = json!([true, false, false, true]);
            }
            entry
        })
        .collect();
    json!({
        "layers": [{"name": "Base", "faces": base}, {"name": "Detail", "faces": detail}],
        "active": "Detail",
        "render": "Base"
    })
}

/// A textured box `body` (size 2) whose faces are `0..6`, carrying [`textured_store`].
fn textured_box() -> SceneDoc {
    let mut doc = apply(
        &empty_scene(),
        json!([{"op":"node.create","id":"body","kind":"box","params":{"size":2}}]),
    )
    .unwrap();
    let ids = face_ids(&doc, "body");
    assert_eq!(ids, [0, 1, 2, 3, 4, 5], "box faces are numbered 0 to 5");
    set_store(&mut doc, "body", textured_store(&ids));
    doc
}

#[test]
fn mesh_delete_drops_rows_of_removed_faces_and_keeps_survivors() {
    let doc = textured_box();
    let after = apply(
        &doc,
        json!([{"op":"mesh.delete","target":{"id":"body"},"elements":{"domain":"face","ids":["f0"]}}]),
    )
    .unwrap();
    assert_eq!(node_store(&after, "body"), textured_store(&[1, 2, 3, 4, 5]));
    after.validate().unwrap();
}

#[test]
fn mesh_separate_selection_prunes_both_outputs_and_keeps_survivors() {
    let doc = textured_box();
    let after = apply(
        &doc,
        json!([{
            "op":"mesh.separate",
            "target":{"id":"body"},
            "mode":"selection",
            "elements":{"domain":"face","ids":["f2","f3"]}
        }]),
    )
    .unwrap();
    assert_eq!(
        node_store(&after, "body_separated_1"),
        textured_store(&[2, 3]),
        "the separated part keeps its selected faces, values, and pins"
    );
    assert_eq!(
        node_store(&after, "body"),
        textured_store(&[0, 1, 4, 5]),
        "the remainder keeps its faces, values, and pins"
    );
    after.validate().unwrap();
}

#[test]
fn rename_touches_only_items_the_edited_mesh_reaches() {
    let mut doc = two_lit_boxes();
    textured_material(&mut doc, "surface", "Light");
    textured_material(&mut doc, "hair", "Light");
    attach(&mut doc, "a", "hair");
    attach(&mut doc, "b", "surface");

    let renamed = apply(
        &doc,
        json!([{"op":"uv.layer_rename","target":{"id":"a"},"name":"Light","to":"Bake"}]),
    )
    .unwrap();
    assert_eq!(layer_names(&renamed, "a"), ["Bake"]);
    assert_eq!(layer_names(&renamed, "b"), ["Light"]);
    assert_eq!(texture_layer(&renamed, "hair").as_deref(), Some("Bake"));
    assert_eq!(texture_layer(&renamed, "surface").as_deref(), Some("Light"));
    assert_eq!(modifier_param(&renamed, "a", "uv_layer"), json!("Bake"));
    assert_eq!(modifier_param(&renamed, "b", "uv_layer"), json!("Light"));
}

#[test]
fn delete_is_refused_only_by_referrers_the_mesh_reaches() {
    let mut doc = two_lit_boxes();
    textured_material(&mut doc, "surface", "Light");
    textured_material(&mut doc, "hair", "Light");
    attach(&mut doc, "a", "hair");
    attach(&mut doc, "b", "surface");

    let refused = apply(
        &doc,
        json!([{"op":"uv.layer_delete","target":{"id":"a"},"name":"Light"}]),
    )
    .unwrap_err();
    assert_eq!(refused.code, ErrorCode::InvalidOperation);
    assert_eq!(refused.details["feature_id"], json!("uv.layer_in_use"));
    let referrers = refused.details["referrers"].as_array().unwrap();
    assert!(
        referrers
            .iter()
            .any(|referrer| referrer.as_str().unwrap().contains("material `hair`"))
    );
    assert!(
        referrers
            .iter()
            .all(|referrer| !referrer.as_str().unwrap().contains("surface")),
        "b's material is not reached by a and must not block the delete"
    );

    let mut unreached = doc.clone();
    unreached
        .nodes
        .get_mut(&Id::new("a").unwrap())
        .unwrap()
        .materials
        .clear();
    unreached
        .nodes
        .get_mut(&Id::new("a").unwrap())
        .unwrap()
        .modifiers
        .clear();
    let deleted = apply(
        &unreached,
        json!([{"op":"uv.layer_delete","target":{"id":"a"},"name":"Light"}]),
    )
    .unwrap();
    assert_eq!(layer_names(&deleted, "a"), Vec::<String>::new());
    assert_eq!(layer_names(&deleted, "b"), ["Light"]);
    assert_eq!(texture_layer(&deleted, "surface").as_deref(), Some("Light"));
}

#[test]
fn rename_of_a_material_shared_with_an_unedited_node_is_rejected_atomically() {
    let mut doc = two_lit_boxes();
    textured_material(&mut doc, "shared", "Light");
    attach(&mut doc, "a", "shared");
    attach(&mut doc, "b", "shared");

    let refused = apply(
        &doc,
        json!([
            {"op":"uv.layer_create","target":{"id":"a"},"name":"Extra"},
            {"op":"uv.layer_rename","target":{"id":"a"},"name":"Light","to":"Bake"}
        ]),
    )
    .unwrap_err();
    assert_eq!(refused.code, ErrorCode::InvalidOperation);
    assert_eq!(refused.details["feature_id"], json!("uv.layer_in_use"));
    assert!(
        refused.details["referrers"]
            .as_array()
            .unwrap()
            .iter()
            .any(|referrer| referrer == "material `shared` is also used by node `b`")
    );
    assert_eq!(
        layer_names(&doc, "a"),
        ["Light"],
        "the source document is untouched"
    );

    doc.nodes
        .get_mut(&Id::new("b").unwrap())
        .unwrap()
        .materials
        .clear();
    let renamed = apply(
        &doc,
        json!([{"op":"uv.layer_rename","target":{"id":"a"},"name":"Light","to":"Bake"}]),
    )
    .unwrap();
    assert_eq!(texture_layer(&renamed, "shared").as_deref(), Some("Bake"));
    assert_eq!(layer_names(&renamed, "b"), ["Light"]);
}

#[test]
fn shared_scope_rename_reaches_every_user_of_the_mesh() {
    let mut doc = two_lit_boxes();
    doc = apply(
        &doc,
        json!([{"op":"node.duplicate","target":{"id":"a"},"id":"a_copy","mode":"linked"}]),
    )
    .unwrap();
    let renamed = apply(
        &doc,
        json!([{"op":"uv.layer_rename","target":{"id":"a"},"scope":"shared","name":"Light","to":"Bake"}]),
    )
    .unwrap();
    assert_eq!(layer_names(&renamed, "a_copy"), ["Bake"]);
    assert_eq!(layer_names(&renamed, "b"), ["Light"]);
}

#[test]
fn layer_named_active_is_not_a_modifier_selector() {
    let mut doc = apply(
        &empty_scene(),
        json!([
            {"op":"node.create","id":"a","kind":"box","params":{"size":1}},
            {"op":"uv.layer_create","target":{"id":"a"},"name":"UVMap"},
            {"op":"uv.layer_create","target":{"id":"a"},"name":"ACTIVE"}
        ]),
    )
    .unwrap();
    add_modifier(
        &mut doc,
        "a",
        "data_transfer",
        json!({"layers_uv_select_src":"ACTIVE","layers_uv_select_dst":"NAME"}),
    );

    let renamed = apply(
        &doc,
        json!([{"op":"uv.layer_rename","target":{"id":"a"},"name":"ACTIVE","to":"Bake"}]),
    )
    .unwrap();
    assert_eq!(layer_names(&renamed, "a"), ["UVMap", "Bake"]);
    assert_eq!(
        modifier_param(&renamed, "a", "layers_uv_select_src"),
        json!("ACTIVE"),
        "the modifier's ALL/ACTIVE/NAME/INDEX selector is not a layer reference"
    );

    let deleted = apply(
        &doc,
        json!([{"op":"uv.layer_delete","target":{"id":"a"},"name":"ACTIVE"}]),
    )
    .unwrap();
    assert_eq!(layer_names(&deleted, "a"), ["UVMap"]);
    assert_eq!(
        modifier_param(&deleted, "a", "layers_uv_select_src"),
        json!("ACTIVE")
    );
}

#[test]
fn layer_rename_schema_publishes_name_and_to() {
    let published = schema("operations", Some("uv.layer_rename")).unwrap();
    let mut fields: Vec<&str> = published["properties"]
        .as_object()
        .unwrap()
        .keys()
        .map(String::as_str)
        .collect();
    fields.sort_unstable();
    assert_eq!(fields, ["name", "op", "scope", "target", "to"]);
    assert_eq!(published["required"], json!(["op", "target", "name", "to"]));
    assert_eq!(published["x-potter"]["target_type"], json!("mesh object"));
    assert_eq!(
        published["x-potter"]["scope"],
        json!("shared|single_user for shared data")
    );
    assert_eq!(published["x-potter"]["defaults"], json!({}));
}

#[test]
fn uv_schemas_do_not_publish_modifier_select_enums_as_layers() {
    for name in OP_NAMES.iter().filter(|name| name.starts_with("uv.")) {
        let text = schema("operations", Some(*name)).unwrap().to_string();
        assert!(!text.contains("layers_uv_select"), "{name}");
    }
}

/// A 16 by 16 canvas and a size-2 box. Its top-face `UVMap` spans [-.5,.5]² after scaling;
/// `Paint` copies those UVs and translates that face to [0,1]².
fn paint_setup() -> Vec<Value> {
    vec![
        json!({"op":"image.create","id":"canvas","width":16,"height":16,"fill_color":[0.0,0.0,0.0,1.0]}),
        json!({"op":"node.create","id":"body","kind":"box","params":{"size":2}}),
        json!({"op":"uv.unwrap","target":{"id":"body"},"method":"box"}),
        json!({"op":"uv.transform","target":{"id":"body"},"scale":[0.5,0.5]}),
        json!({"op":"uv.layer_create","target":{"id":"body"},"name":"Paint"}),
        json!({"op":"uv.transform","target":{"id":"body"},"uv_layer":"Paint","translation":[0.5,0.5]}),
        json!({"op":"uv.layer_set_active","target":{"id":"body"},"name":"Paint"}),
    ]
}

fn paint_operation(uv_layer: Option<&str>) -> Value {
    let mut operation = json!({
        "op":"paint.texture",
        "target":{"id":"body"},
        "image":"canvas",
        "samples":[{"position":[0.0,0.0,1.0],"radius":0.2,"pressure":1.0,"strength":1.0,"time":0.0}],
        "color":[1.0,0.0,0.0,1.0],
        "blend":"mix",
        "falloff":"linear",
        "seed":7
    });
    if let Some(uv_layer) = uv_layer {
        operation["uv_layer"] = json!(uv_layer);
    }
    operation
}

fn paint_image_pixels(outcome: &ApplyOutcome) -> Vec<[f64; 4]> {
    let image = &outcome.doc.images[&Id::new("canvas").unwrap()];
    let digest = image.blob.as_ref().unwrap();
    let bytes = &outcome.asset_blobs[digest];
    potter_core::image::decode_pixels(bytes, image.colorspace)
        .unwrap()
        .2
}

#[test]
fn paint_without_a_selector_samples_the_active_layer() {
    let root = tempdir().unwrap();
    let empty = empty_scene();
    let run = |extra: Option<Value>| {
        let mut operations = paint_setup();
        operations.extend(extra);
        apply_at(&empty, Value::Array(operations), root.path())
    };

    let baseline = run(None).unwrap();
    let omitted = run(Some(paint_operation(None))).unwrap();
    let named_active = run(Some(paint_operation(Some("Paint")))).unwrap();
    let named_render = run(Some(paint_operation(Some("UVMap")))).unwrap();

    // Replaced staged blobs are pruned, so detect the paint by decoding the referenced image pixels.
    let baseline_pixels = paint_image_pixels(&baseline);
    let omitted_pixels = paint_image_pixels(&omitted);
    let changed_pixels = baseline_pixels
        .iter()
        .zip(&omitted_pixels)
        .enumerate()
        .filter_map(|(index, (before, after))| {
            before
                .iter()
                .zip(after)
                .any(|(before, after)| (*before - *after).abs() > 0.0)
                .then_some(index)
        })
        .collect::<Vec<_>>();
    assert_eq!(
        changed_pixels,
        vec![103, 104, 118, 119, 120, 121, 134, 135, 136, 137, 151, 152],
        "the world-space stroke reaches the expected 12 active-layer texels"
    );
    assert_eq!(
        omitted_pixels,
        paint_image_pixels(&named_active),
        "an omitted selector paints the active layer"
    );
    assert_ne!(
        omitted_pixels,
        paint_image_pixels(&named_render),
        "the active and render layers paint different texels"
    );

    let missing = run(Some(paint_operation(Some("Nope")))).unwrap_err();
    assert_eq!(missing.details["feature_id"], json!("uv.layer_missing"));
}

#[test]
fn empty_uv_layer_selector_is_rejected_on_uv_paint_and_texture_ops() {
    let root = tempdir().unwrap();
    let empty = empty_scene();

    let mut pack_setup = paint_setup();
    pack_setup.truncate(3);
    pack_setup.push(json!({"op":"uv.pack","target":{"id":"body"},"uv_layer":""}));
    let pack = apply_at(&empty, Value::Array(pack_setup), root.path()).unwrap_err();
    assert_eq!(pack.code, ErrorCode::InvalidOperation);
    assert_eq!(pack.details["pointer"], json!("/uv_layer"));
    assert_eq!(pack.details["operation_index"], json!(3));

    let mut paint_ops = paint_setup();
    paint_ops.truncate(2);
    paint_ops.push(paint_operation(Some("")));
    let paint = apply_at(&empty, Value::Array(paint_ops), root.path()).unwrap_err();
    assert_eq!(paint.code, ErrorCode::InvalidOperation);
    assert_eq!(paint.details["pointer"], json!("/uv_layer"));
    assert_eq!(paint.details["operation_index"], json!(2));

    let material = apply_at(
        &empty,
        json!([
            {"op":"image.create","id":"canvas","width":2,"height":2,"fill_color":[0.0,0.0,0.0,1.0]},
            {"op":"material.create","id":"surface",
             "base_color_texture":{"image":"canvas","uv_layer":""}}
        ]),
        root.path(),
    )
    .unwrap_err();
    assert_eq!(material.code, ErrorCode::InvalidOperation);
    assert_eq!(material.details["operation_index"], json!(1));
    assert!(
        material.details["pointer"]
            .as_str()
            .unwrap()
            .ends_with("/base_color_texture/uv_layer"),
        "{}",
        material.details["pointer"]
    );
}

#[test]
fn group_reached_only_through_another_mesh_is_not_rewritten() {
    // A group whose UV map node names `Light` is reached only through a material the edited mesh
    // uses. Renaming the layer on a mesh that does not use that material leaves the group alone.
    let mut doc = two_lit_boxes();
    let mut graph = NodeGroup::new("Surface", GraphKind::Shader);
    let mut uv_node = GraphNode::new("ShaderNodeUVMap");
    uv_node
        .properties
        .insert("uv_map".to_owned(), json!("Light"));
    graph.nodes.insert(Id::new("uv").unwrap(), uv_node);
    doc.node_groups
        .insert(Id::new("surface_graph").unwrap(), graph);
    textured_material(&mut doc, "surface", "Light");
    doc.materials
        .get_mut(&Id::new("surface").unwrap())
        .unwrap()
        .node_tree = Some(Id::new("surface_graph").unwrap());
    attach(&mut doc, "b", "surface");

    let renamed = apply(
        &doc,
        json!([{"op":"uv.layer_rename","target":{"id":"a"},"name":"Light","to":"Bake"}]),
    )
    .unwrap();
    assert_eq!(
        renamed.node_groups[&Id::new("surface_graph").unwrap()].nodes[&Id::new("uv").unwrap()]
            .properties["uv_map"],
        json!("Light")
    );
}

#[test]
fn layer_reached_through_a_geometry_modifier_node_group_is_scoped_like_a_material_graph() {
    // The `Light` UV map sits in a shader group that a group node of the geometry modifier's node
    // group instances. The edited mesh reaches it through that modifier, so renaming the layer
    // rewrites it and deleting the layer is refused by it.
    let mut doc = two_lit_boxes();
    let mut inner = NodeGroup::new("Inner", GraphKind::Shader);
    let mut uv_node = GraphNode::new("ShaderNodeUVMap");
    uv_node
        .properties
        .insert("uv_map".to_owned(), json!("Light"));
    inner.nodes.insert(Id::new("uv").unwrap(), uv_node);
    doc.node_groups
        .insert(Id::new("inner_graph").unwrap(), inner);
    let mut geometry = NodeGroup::new("Geometry", GraphKind::Geometry);
    let mut group_node = GraphNode::new("ShaderNodeGroup");
    group_node
        .properties
        .insert("node_tree".to_owned(), json!({"name": "Inner"}));
    geometry.nodes.insert(Id::new("group").unwrap(), group_node);
    doc.node_groups
        .insert(Id::new("geometry_graph").unwrap(), geometry);
    add_modifier(
        &mut doc,
        "a",
        "nodes",
        json!({"node_group": "geometry_graph"}),
    );

    let renamed = apply(
        &doc,
        json!([{"op":"uv.layer_rename","target":{"id":"a"},"name":"Light","to":"Bake"}]),
    )
    .unwrap();
    assert_eq!(
        renamed.node_groups[&Id::new("inner_graph").unwrap()].nodes[&Id::new("uv").unwrap()]
            .properties["uv_map"],
        json!("Bake")
    );

    let refused = apply(
        &doc,
        json!([{"op":"uv.layer_delete","target":{"id":"a"},"name":"Light"}]),
    )
    .unwrap_err();
    assert_eq!(refused.details["feature_id"], json!("uv.layer_in_use"));
    assert!(
        refused.details["referrers"]
            .as_array()
            .unwrap()
            .iter()
            .any(|referrer| referrer
                == "ShaderNodeUVMap node `uv` in node group `inner_graph` property `uv_map`")
    );
}
