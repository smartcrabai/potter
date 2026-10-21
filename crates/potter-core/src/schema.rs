use serde_json::{Map, Value, json};

use crate::{
    catalog,
    error::{ErrorCode, PotError, Result},
    ops,
    params::{self, ParameterFamily},
};

const DRAFT: &str = "https://json-schema.org/draft/2020-12/schema";

pub fn schema(kind: &str, operation: Option<&str>) -> Result<Value> {
    if let Some(operation) = operation {
        if kind != "operations" {
            return Err(PotError::new(
                ErrorCode::InvalidArgument,
                "--op is only valid with --kind operations",
            ));
        }
        if !ops::OP_NAMES.contains(&operation) {
            return Err(PotError::with_details(
                ErrorCode::InvalidArgument,
                "unknown operation name",
                json!({"op":operation}),
            ));
        }
        return Ok(operation_schema(operation));
    }
    match kind {
        "scene" => Ok(scene_schema()),
        "operations" => Ok(operations_schema()),
        "preview" => Ok(preview_schema()),
        "response" => Ok(response_schema()),
        "capabilities" => Ok(catalog::capabilities_schema()),
        "formats" => Ok(catalog::formats_schema()),
        _ => Err(PotError::with_details(
            ErrorCode::InvalidArgument,
            "unknown schema kind",
            json!({"kind":kind}),
        )),
    }
}

fn base(title: &str) -> Value {
    json!({"$schema":DRAFT,"title":title,"type":"object"})
}

fn id_schema() -> Value {
    json!({"type":"string","pattern":"^[a-z][a-z0-9_-]{0,63}$"})
}
fn nullable_id_array_schema() -> Value {
    json!({"type":"array","items":{"oneOf":[{"$ref":"#/$defs/id"},{"type":"null"}]}})
}

fn vec_schema(size: usize) -> Value {
    json!({"type":"array","items":{"type":"number"},"minItems":size,"maxItems":size})
}

fn keyframe_schema() -> Value {
    let bezier_handle = json!({
        "type":["array", "null"],
        "items":{"type":"number"},
        "minItems":2,
        "maxItems":2
    });
    json!({
        "type":"object",
        "required":["frame", "value"],
        "properties":{
            "frame":{"type":"number"},
            "value":{"type":"number"},
            "interpolation":{"enum":["constant", "linear", "bezier"], "default":"linear"},
            "handle_left":bezier_handle,
            "handle_right":bezier_handle
        },
        "additionalProperties":false
    })
}

fn fcurve_schema() -> Value {
    json!({"type":"object","required":["path","index"],"properties":{"path":{"type":"string"},"index":{"type":"integer","minimum":0,"maximum":u32::MAX},"keyframes":{"type":"array","items":{"$ref":"#/$defs/keyframe"},"default":[]},"extrapolation":{"enum":["constant","linear"],"default":"constant"}},"additionalProperties":false})
}

fn action_schema() -> Value {
    json!({"type":"object","required":["name"],"properties":{"name":{"type":"string"},"fcurves":{"type":"array","items":{"$ref":"#/$defs/fcurve"},"default":[]}},"additionalProperties":false})
}

fn scene_schema() -> Value {
    let mut root = base("Potter scene document");
    root["required"] = json!(["schema_version", "scene_id", "revision", "active_scene"]);
    root["properties"] = json!({
        "schema_version":{"const":7},
        "scene_id":{"type":"string","format":"uuid"},
        "revision":{"type":"integer","minimum":0,"maximum":9_007_199_254_740_991_u64},
        "active_scene":{"$ref":"#/$defs/id"},
        "profile":{"$ref":"#/$defs/profile"},
        "scenes":registry_schema("scene"),
        "collections":registry_schema("collection"),
        "nodes":registry_schema("node"),
        "data_blocks":registry_schema("data_block"),
        "materials":registry_schema("material"),
        "worlds":registry_schema("world"),
        "node_groups":{"type":"object"},
        "actions":registry_schema("action"),
        "resources":{"type":"object","additionalProperties":{"type":"object","properties":{"storage":{"const":"inline"},"kind":{"type":"string"}},"allOf":[{"if":{"required":["storage"]},"then":{"required":["kind"],"allOf":[{"if":{"properties":{"kind":{"const":"legacy_texture"}},"required":["kind"]},"then":{"$ref":"#/$defs/legacy_texture_resource"}}]}}]}},
        "libraries":registry_schema("library"),
        "compatibility":{"type":"object"},
        "avatar":avatar_settings_schema(),
        "history":{"$ref":"#/$defs/history"}
    });
    root["additionalProperties"] = json!(false);
    // The scene document requires `interpolation`; the operation schema defaults it to linear.
    let mut texture_ref = texture_ref_schema();
    texture_ref["required"] = json!(["image", "interpolation"]);
    root["$defs"] = json!({
        "legacy_texture_resource":{"type":"object","required":["storage","kind","texture_type"],"properties":{"storage":{"const":"inline"},"kind":{"const":"legacy_texture"},"texture_type":{"enum":["CLOUDS","MARBLE","WOOD","MAGIC","BLEND","STUCCI","NOISE","DISTORTED_NOISE","VORONOI","MUSGRAVE","IMAGE"]},"settings":{"type":"object","properties":{"noise_scale":{"type":"number"},"noise_depth":{"type":"integer"},"noise_basis":{"type":"string"},"noise_type":{"type":"string"},"noise_basis_2":{"type":"string"},"turbulence":{"type":"number"},"cloud_type":{"type":"string"},"wood_type":{"type":"string"},"marble_type":{"type":"string"},"stucci_type":{"type":"string"},"distortion":{"type":"number"},"musgrave_type":{"type":"string"},"dimension_max":{"type":"number"},"lacunarity":{"type":"number"},"offset":{"type":"number"},"gain":{"type":"number"},"noise_intensity":{"type":"number"},"weight_1":{"type":"number"},"weight_2":{"type":"number"},"weight_3":{"type":"number"},"weight_4":{"type":"number"},"minkovsky_exponent":{"type":"number"},"distance_metric":{"type":"string"},"color_mode":{"type":"string"},"noise_distortion":{"type":"string"},"progression":{"enum":["LINEAR","QUADRATIC","EASING","DIAGONAL","SPHERICAL","QUADRATIC_SPHERE","RADIAL"]},"use_flip_axis":{"enum":["HORIZONTAL","VERTICAL"]}},"additionalProperties":false}},"additionalProperties":false},
        "id":id_schema(),
        "vec2":vec_schema(2),
        "vec3":vec_schema(3),
        "profile":{"type":"object","required":["blender"],"properties":{"blender":{"type":"string"}},"additionalProperties":false},
        "unit":{"type":"object","required":["system","scale_length"],"properties":{"system":{"type":"string"},"scale_length":{"type":"number","exclusiveMinimum":0}},"additionalProperties":false},
        "view_layer":{"type":"object","required":["name","excluded_collections"],"properties":{"name":{"type":"string"},"excluded_collections":{"type":"array","items":{"$ref":"#/$defs/id"}}},"additionalProperties":false},
        "scene":{"type":"object","required":["name","root_collection","view_layers","frame_current","frame_start","frame_end","fps","fps_base","camera","world","unit","markers","render","rigid_body_world"],"properties":{"name":{"type":"string"},"root_collection":{"$ref":"#/$defs/id"},"view_layers":registry_schema("view_layer"),"frame_current":{"type":"number"},"frame_start":{"type":"integer"},"frame_end":{"type":"integer"},"fps":{"type":"integer","minimum":1},"fps_base":{"type":"number","exclusiveMinimum":0},"camera":{"oneOf":[{"$ref":"#/$defs/id"},{"type":"null"}]},"world":{"oneOf":[{"$ref":"#/$defs/id"},{"type":"null"}]},"unit":{"$ref":"#/$defs/unit"},"render":{"$ref":"#/$defs/render_settings"},"markers":{"type":"array","items":{"$ref":"#/$defs/marker"}},"rigid_body_world":{"oneOf":[{"$ref":"#/$defs/rigid_body_world"},{"type":"null"}]}},"additionalProperties":false},
        "render_settings":{"type":"object","required":["resolution_x","resolution_y","resolution_percentage","samples","seed","max_bounces","film_transparent","engine","use_sequencer","audio_codec","passes"],"properties":{"resolution_x":{"type":"integer","minimum":1,"maximum":16384},"resolution_y":{"type":"integer","minimum":1,"maximum":16384},"resolution_percentage":{"type":"integer","minimum":1,"maximum":100},"samples":{"type":"integer","minimum":1,"maximum":1_000_000},"seed":{"type":"integer","minimum":0,"maximum":u32::MAX},"max_bounces":{"type":"integer","minimum":0,"maximum":1024},"transparent_max_bounces":{"type":"integer","minimum":0,"maximum":1024,"default":8},"film_transparent":{"type":"boolean"},"engine":{"enum":["path","realtime"]},"use_sequencer":{"type":"boolean"},"audio_codec":{"enum":["wav","flac"]},"passes":{"type":"array","items":{"type":"string"}},"cryptomatte_accurate":{"type":"boolean"}},"additionalProperties":false},
        "marker":{"type":"object","required":["id","name","frame"],"properties":{"id":{"$ref":"#/$defs/id"},"name":{"type":"string"},"frame":{"type":"number"}},"additionalProperties":false},
        "rigid_body_world":{"type":"object","properties":{"enabled":{"type":"boolean"},"gravity":{"$ref":"#/$defs/vec3"},"substeps":{"type":"integer","minimum":1,"maximum":128},"solver_iterations":{"type":"integer","minimum":1,"maximum":128},"frame_start":{"type":"integer","minimum":i32::MIN,"maximum":i32::MAX},"frame_end":{"type":"integer","minimum":i32::MIN,"maximum":i32::MAX},"seed":{"type":"integer","minimum":0,"maximum":u32::MAX}},"additionalProperties":false},
        "rigid_body":{"type":"object","properties":{"type":{"enum":["active","passive"]},"mass":{"type":"number","minimum":0},"friction":{"type":"number","minimum":0},"restitution":{"type":"number","minimum":0,"maximum":1},"shape":{"enum":["box","sphere","convex_hull","mesh"]},"linear_damping":{"type":"number","minimum":0},"angular_damping":{"type":"number","minimum":0},"initial_velocity":{"$ref":"#/$defs/vec3"}},"additionalProperties":false},
        "force_field":{"type":"object","properties":{"type":{"enum":["wind","vortex","force"]},"strength":{"type":"number"},"falloff":{"type":"number","minimum":0}},"additionalProperties":false},
        "collection":{"type":"object","required":["name","children","objects"],"properties":{"name":{"type":"string"},"children":{"type":"array","items":{"$ref":"#/$defs/id"}},"objects":{"type":"array","items":{"$ref":"#/$defs/id"}}},"additionalProperties":false},
        "transform":{"type":"object","required":["translation","rotation","scale","rotation_mode"],"properties":{"translation":{"$ref":"#/$defs/vec3"},"rotation":{"type":"array","items":{"type":"number"},"minItems":4,"maxItems":4},"scale":{"$ref":"#/$defs/vec3"},"rotation_mode":{"type":"string"}},"additionalProperties":false},
        "modifier":{"type":"object","required":["id","type","name","enabled","params"],"properties":{"id":{"$ref":"#/$defs/id"},"type":{"type":"string"},"name":{"type":"string"},"enabled":{"type":"boolean"},"show_viewport":{"type":"boolean"},"show_render":{"type":"boolean"},"params":{"type":"object"}},"additionalProperties":false},
        "library":{"type":"object","required":["name","kind","uri","resolved_path","resource","hash","status","linked_ids","overrides","items","source_project"],"properties":{"name":{"type":"string"},"kind":{"enum":["potter_project","blend"]},"uri":{"type":"string"},"resolved_path":{"type":"string"},"resource":{"oneOf":[{"$ref":"#/$defs/id"},{"type":"null"}]},"hash":{"type":"string"},"status":{"enum":["ok","missing","changed"]},"linked_ids":{"type":"object","propertyNames":{"enum":["nodes","data_blocks","materials","collections","actions","node_groups","worlds","resources"]},"additionalProperties":{"type":"array","items":{"$ref":"#/$defs/id"}}},"overrides":{"type":"array","items":{"$ref":"#/$defs/library_override"}},"items":{"type":"object","additionalProperties":{"type":"string"}},"source_project":{"type":["string","null"]}},"additionalProperties":false},
        "library_override":{"type":"object","required":["registry","id","reference_id","properties"],"properties":{"registry":{"type":"string"},"id":{"$ref":"#/$defs/id"},"reference_id":{"$ref":"#/$defs/id"},"properties":{"type":"array","items":{"$ref":"#/$defs/library_override_property"}}},"additionalProperties":false},
        "library_override_property":{"type":"object","required":["path","operation","value"],"properties":{"path":{"type":"string"},"operation":{"enum":["replace","insert_after","delete"]},"value":{}},"additionalProperties":false},
        "node":{"type":"object","required":["name","kind"],"properties":{"name":{"type":"string"},"kind":{"enum":["empty","mesh","camera","light"]},"primitive":{"type":["string","null"]},"tags":{"type":"array","items":{"$ref":"#/$defs/id"}},"parent":{"oneOf":[{"$ref":"#/$defs/id"},{"type":"null"}]},"parent_inverse":{"oneOf":[{"type":"array","items":{"type":"number"},"minItems":16,"maxItems":16},{"type":"null"}]},"transform":{"$ref":"#/$defs/transform"},"data":{"oneOf":[{"$ref":"#/$defs/id"},{"type":"null"}]},"materials":{"type":"array","items":{"oneOf":[{"$ref":"#/$defs/id"},{"type":"null"}]}},"modifiers":{"type":"array","items":{"$ref":"#/$defs/modifier"}},"visible":{"type":"boolean"},"render_visible":{"type":"boolean"},"selectable":{"type":"boolean"},"action":{"oneOf":[{"$ref":"#/$defs/id"},{"type":"null"}]},"nla_tracks":{"type":"array","items":{"type":"object"}},"properties":{"type":"object"},"rigid_body":{"oneOf":[{"$ref":"#/$defs/rigid_body"},{"type":"null"}]},"force_field":{"oneOf":[{"$ref":"#/$defs/force_field"},{"type":"null"}]},"parent_type":{"enum":["object","bone"]},"parent_bone":{"oneOf":[{"$ref":"#/$defs/id"},{"type":"null"}]},"pose":{"type":"object","additionalProperties":{"type":"object"}},"constraints":{"type":"array","items":{"type":"object"}},"drivers":{"type":"array","items":{"type":"object"}}},"additionalProperties":false},
        "vertex":{"type":"object","required":["id","co"],"properties":{"id":{"type":"integer","minimum":0,"maximum":u32::MAX},"co":{"$ref":"#/$defs/vec3"}},"additionalProperties":false},
        "edge":{"type":"object","required":["id","v"],"properties":{"id":{"type":"integer","minimum":0,"maximum":u32::MAX},"v":{"type":"array","items":{"type":"integer","minimum":0,"maximum":u32::MAX},"minItems":2,"maxItems":2}},"additionalProperties":false},
        "face":{"type":"object","required":["id","v","material_index"],"properties":{"id":{"type":"integer","minimum":0,"maximum":u32::MAX},"v":{"type":"array","items":{"type":"integer","minimum":0,"maximum":u32::MAX},"minItems":3},"material_index":{"type":"integer","minimum":0,"maximum":u32::MAX}},"additionalProperties":false},
        "mesh":{"type":"object","required":["vertices","edges","faces","attributes","next_id"],"properties":{"vertices":{"type":"array","items":{"$ref":"#/$defs/vertex"}},"edges":{"type":"array","items":{"$ref":"#/$defs/edge"}},"faces":{"type":"array","items":{"$ref":"#/$defs/face"}},"attributes":{"type":"object"},"next_id":{"type":"object","properties":{"vertex":{"type":"integer","minimum":0},"edge":{"type":"integer","minimum":0},"face":{"type":"integer","minimum":0}},"required":["vertex","edge","face"],"additionalProperties":false}},"additionalProperties":false},
        "data_block":{"type":"object","required":["type"],"properties":{"type":{"type":"string"},"descriptor":{"type":["object","null"]},"mesh":{"oneOf":[{"$ref":"#/$defs/mesh"},{"type":"null"}]},"camera":{"oneOf":[{"$ref":"#/$defs/camera"},{"type":"null"}]},"light":{"oneOf":[{"$ref":"#/$defs/light"},{"type":"null"}]},"grease_pencil":{"type":["object","null"]},"armature":{"type":["object","null"]},"shape_keys":{"type":["object","null"]},"vertex_groups":{"type":"array","items":{"type":"object"}},"vertex_weights":{"type":"object"}},"additionalProperties":false},
        "camera": camera_data_schema(),
        "light":{"type":"object","required":["light_type","color","energy","radius","spot_size","spot_blend"],"properties":{"light_type":{"enum":["point","sun","spot","area"]},"color":{"$ref":"#/$defs/vec3"},"energy":{"type":"number"},"radius":{"type":"number"},"soft_falloff":{"type":"boolean"},"use_multiple_importance_sampling":{"type":"boolean"},"spread":{"type":"number","minimum":0,"maximum":std::f64::consts::PI},"spot_size":{"type":"number"},"spot_blend":{"type":"number"}},"additionalProperties":false},
        "material":{"type":"object","required":["name","base_color","metallic","roughness","emission_color","emission_strength","transmission","ior","double_sided","node_tree","base_color_texture","roughness_texture","metallic_texture","normal_texture"],"properties":{"name":{"type":"string"},"base_color":{"type":"array","items":{"type":"number","minimum":0,"maximum":1},"minItems":4,"maxItems":4},"metallic":{"type":"number","minimum":0,"maximum":1},"roughness":{"type":"number","minimum":0,"maximum":1},"emission_color":{"$ref":"#/$defs/vec3"},"emission_strength":{"type":"number","minimum":0},"transmission":{"type":"number","minimum":0,"maximum":1},"ior":{"type":"number","exclusiveMinimum":0},"double_sided":{"type":"boolean"},"node_tree":{"oneOf":[{"$ref":"#/$defs/id"},{"type":"null"}]},"base_color_texture":{"oneOf":[{"$ref":"#/$defs/texture_ref"},{"type":"null"}]},"roughness_texture":{"oneOf":[{"$ref":"#/$defs/texture_ref"},{"type":"null"}]},"metallic_texture":{"oneOf":[{"$ref":"#/$defs/texture_ref"},{"type":"null"}]},"normal_texture":{"oneOf":[{"$ref":"#/$defs/texture_ref"},{"type":"null"}]}},"additionalProperties":false},
        "texture_ref":texture_ref,
        "world":{"type":"object","required":["color","strength"],"properties":{"color":{"$ref":"#/$defs/vec3"},"strength":{"type":"number","minimum":0}},"additionalProperties":false},
        "keyframe":keyframe_schema(),
        "fcurve":fcurve_schema(),
        "action":action_schema(),
        "history":{"type":"object","properties":{"head":{"type":["string","null"]},"undo":{"type":"array","items":{"type":"string"}},"redo":{"type":"array","items":{"type":"string"}}},"additionalProperties":false}
    });
    root["$defs"]["constraint"] = json!({
        "type":"object",
        "required":["id","type","name","target","subtarget","owner_bone","influence","enabled","params"],
        "properties":{
            "id":{"$ref":"#/$defs/id"},
            "type":{"enum":[
                "copy_location","copy_rotation","copy_scale","track_to","damped_track",
                "locked_track","stretch_to","transformation","maintain_volume","floor",
                "pivot","shrinkwrap","spline_ik","limit_location","limit_rotation",
                "limit_scale","child_of","action","armature","camera_solver","clamp_to",
                "copy_transforms","follow_path","follow_track","geometry_attribute",
                "limit_distance","object_solver","transform_cache","ik"
            ]},
            "name":{"type":"string"},
            "target":{"oneOf":[{"$ref":"#/$defs/id"},{"type":"null"}]},
            "subtarget":{"oneOf":[{"$ref":"#/$defs/id"},{"type":"null"}]},
            "owner_bone":{"oneOf":[{"$ref":"#/$defs/id"},{"type":"null"}]},
            "influence":{"type":"number","minimum":0,"maximum":1},
            "enabled":{"type":"boolean"},
            "params":{"type":"object"},
            "inverse_matrix":{"oneOf":[{"type":"array","items":{"type":"number"},"minItems":16,"maxItems":16},{"type":"null"}]},
        },
        "additionalProperties":false
    });
    root["$defs"]["node"]["properties"]["constraints"]["items"] =
        json!({"$ref":"#/$defs/constraint"});
    root["$defs"]["node"]["properties"]["materials"] = nullable_id_array_schema();
    root["$defs"]["node"]["properties"]["material_links"] = json!({
        "type":"array",
        "items":{"enum":["data","object"]}
    });
    root["$defs"]["node"]["properties"]["track_axis"] =
        json!({"enum":["POS_X","POS_Y","POS_Z","NEG_X","NEG_Y","NEG_Z"]});
    root["$defs"]["node"]["properties"]["up_axis"] = json!({"enum":["X","Y","Z"]});
    root["$defs"]["captured_matrix_basis"] = json!({
        "type":"object",
        "required":["matrix_basis","fingerprint"],
        "properties":{
            "matrix_basis":{"type":"array","items":{"type":"number"},"minItems":16,"maxItems":16},
            "fingerprint":{"type":"string"}
        },
        "additionalProperties":false
    });
    root["$defs"]["node"]["properties"]["blender_matrix_basis"] =
        json!({"oneOf":[{"$ref":"#/$defs/captured_matrix_basis"},{"type":"null"}]});
    root["$defs"]["data_block"]["properties"]["materials"] = nullable_id_array_schema();
    root["$defs"]["material"]["properties"]["viewport_display"] = json!({
        "type":"object",
        "required":["diffuse_color","roughness","metallic"],
        "properties":{"diffuse_color":vec_schema(4),"roughness":{"type":"number","minimum":0,"maximum":1},"metallic":{"type":"number","minimum":0,"maximum":1}},
        "additionalProperties":false
    });
    complete_scene_schema(&mut root);
    root
}

fn complete_scene_schema(root: &mut Value) {
    root["properties"]["images"] = registry_schema("image");
    root["properties"]["movie_clips"] = registry_schema("movie_clip");
    root["properties"]["masks"] = registry_schema("mask");
    let defs = &mut root["$defs"];
    defs["legacy_texture_resource"]["properties"]["texture_type"]["enum"] = json!([
        "CLOUDS",
        "MARBLE",
        "WOOD",
        "MAGIC",
        "BLEND",
        "STUCCI",
        "NOISE",
        "DISTORTED_NOISE",
        "VORONOI",
        "MUSGRAVE",
        "NATIVE_VALUE_NOISE",
        "IMAGE"
    ]);
    defs["legacy_texture_resource"]["properties"]["settings"]["properties"]["octaves"] =
        json!({"type":"number"});
    defs["legacy_texture_resource"]["properties"]["settings"]["properties"]["color_mode"] =
        json!({"enum":["INTENSITY","POSITION","POSITION_OUTLINE","POSITION_OUTLINE_INTENSITY"]});
    defs["legacy_texture_resource"]["allOf"] = json!([{
        "if":{"properties":{"texture_type":{"const":"MUSGRAVE"}},"required":["texture_type"]},
        "then":{"properties":{"settings":{"properties":{
            "octaves":{"type":"number","minimum":0,"maximum":8},
            "lacunarity":{"type":"number","minimum":0,"maximum":6},
            "offset":{"type":"number","minimum":0,"maximum":6},
            "gain":{"type":"number","minimum":0,"maximum":6},
            "dimension_max":{"type":"number","minimum":0.0001,"maximum":2},
            "noise_intensity":{"type":"number","minimum":0,"maximum":10}
        }}}}
    }]);

    defs["image_tile"] = json!({"type":"object","required":["number","width","height","blob"],"properties":{"number":{"type":"integer","minimum":1,"maximum":u32::MAX},"width":{"type":"integer","minimum":0,"maximum":u32::MAX},"height":{"type":"integer","minimum":0,"maximum":u32::MAX},"blob":{"type":["string","null"]},"label":{"type":"string"}},"additionalProperties":false});
    defs["image"] = json!({"type":"object","required":["name","source","colorspace","width","height","tiles","blob","source_path","source_hash","alpha_mode"],"properties":{"name":{"type":"string"},"source":{"enum":["generated","file","packed"]},"colorspace":{"enum":["srgb","linear","non_color"]},"width":{"type":"integer","minimum":0,"maximum":u32::MAX},"height":{"type":"integer","minimum":0,"maximum":u32::MAX},"tiles":{"type":"array","items":{"$ref":"#/$defs/image_tile"}},"blob":{"type":["string","null"]},"source_path":{"type":["string","null"]},"source_hash":{"type":["string","null"]},"alpha_mode":{"enum":["straight","premultiplied","none"]},"tiled":{"type":"boolean"}},"additionalProperties":false});
    defs["mask"] = json!({"type":"object","required":["splines"],"properties":{"splines":{"type":"array","items":{"$ref":"#/$defs/mask_spline"}}},"additionalProperties":false});
    defs["mask_spline"] = json!({"type":"object","required":["points","feather"],"properties":{"points":{"type":"array","minItems":3,"items":{"$ref":"#/$defs/mask_control_point"}},"feather":{"type":"number","minimum":0}},"additionalProperties":false});
    defs["mask_control_point"] = json!({"type":"object","required":["position","keyframes"],"properties":{"position":{"$ref":"#/$defs/vec2"},"keyframes":{"type":"array","items":{"type":"object","required":["frame","position"],"properties":{"frame":{"type":"number"},"position":{"$ref":"#/$defs/vec2"}},"additionalProperties":false}}},"additionalProperties":false});
    defs["movie_clip"] = json!({"type":"object","required":["name","source","source_hash","frame_start","width","height","fps","tracking"],"properties":{"name":{"type":"string"},"source":{"type":["string","null"]},"source_hash":{"type":["string","null"]},"frame_start":{"type":"integer","minimum":i32::MIN,"maximum":i32::MAX},"width":{"type":"integer","minimum":0,"maximum":u32::MAX},"height":{"type":"integer","minimum":0,"maximum":u32::MAX},"fps":{"type":"number","exclusiveMinimum":0},"tracking":{"$ref":"#/$defs/movie_tracking"}},"additionalProperties":false});
    defs["movie_tracking"] = json!({"type":"object","required":["tracks","plane_tracks","camera","reconstruction","objects"],"properties":{"tracks":{"type":"array","items":{"type":"object","required":["id","name","markers"],"properties":{"id":{"type":"string"},"name":{"type":"string"},"markers":{"type":"array","items":{"type":"object","required":["frame","co","pattern_corners","search_area","disabled"],"properties":{"frame":{"type":"number"},"co":{"$ref":"#/$defs/vec2"},"pattern_corners":{"type":"array","items":{"$ref":"#/$defs/vec2"},"minItems":4,"maxItems":4},"search_area":{"type":"array","items":{"$ref":"#/$defs/vec2"},"minItems":4,"maxItems":4},"disabled":{"type":"boolean"}},"additionalProperties":false}}},"additionalProperties":false}},"plane_tracks":{"type":"array","items":{"type":"object","required":["id","name","track_ids","reference_frame","homographies"],"properties":{"id":{"type":"string"},"name":{"type":"string"},"track_ids":{"type":"array","items":{"type":"string"}},"reference_frame":{"type":"number"},"homographies":{"type":"array","items":{"type":"object","required":["frame","matrix"],"properties":{"frame":{"type":"number"},"matrix":{"type":"array","items":{"$ref":"#/$defs/vec3"},"minItems":3,"maxItems":3}},"additionalProperties":false}}},"additionalProperties":false}},"camera":{"type":"object","required":["focal_mm","sensor_width_mm","principal","units","pixel_aspect","distortion_model","k1","k2","k3","division_k1","division_k2","nuke_k1","nuke_k2","nuke_p1","nuke_p2","brown_k1","brown_k2","brown_k3","brown_k4","brown_p1","brown_p2"],"properties":{"focal_mm":{"type":"number"},"sensor_width_mm":{"type":"number"},"principal":{"$ref":"#/$defs/vec2"},"units":{"enum":["PIXELS","MILLIMETERS"]},"pixel_aspect":{"type":"number"},"distortion_model":{"enum":["POLYNOMIAL","DIVISION","NUKE","BROWN"]},"k1":{"type":"number"},"k2":{"type":"number"},"k3":{"type":"number"},"division_k1":{"type":"number"},"division_k2":{"type":"number"},"nuke_k1":{"type":"number"},"nuke_k2":{"type":"number"},"nuke_p1":{"type":"number"},"nuke_p2":{"type":"number"},"brown_k1":{"type":"number"},"brown_k2":{"type":"number"},"brown_k3":{"type":"number"},"brown_k4":{"type":"number"},"brown_p1":{"type":"number"},"brown_p2":{"type":"number"}},"additionalProperties":false},"reconstruction":{"type":"object","required":["cameras","points","is_valid","average_error"],"properties":{"cameras":{"type":"array","items":{"type":"object","required":["frame","matrix","average_error","matrix_is_camera_to_world"],"properties":{"frame":{"type":"integer","minimum":i32::MIN,"maximum":i32::MAX},"matrix":{"type":"array","items":{"type":"array","items":{"type":"number"},"minItems":4,"maxItems":4},"minItems":3,"maxItems":3},"average_error":{"type":"number"},"matrix_is_camera_to_world":{"type":"boolean"}},"additionalProperties":false}},"points":{"type":"array","items":{"type":"object","required":["track","co"],"properties":{"track":{"type":"string"},"co":{"$ref":"#/$defs/vec3"}},"additionalProperties":false}},"is_valid":{"type":"boolean"},"average_error":{"type":"number"}},"additionalProperties":false},"objects":{"type":"array"}},"additionalProperties":false});
    defs["movie_tracking"]["properties"]["objects"]["items"] = json!({"type":"object","required":["id","name","tracks","reconstruction","reconstruction_is_valid","reconstruction_average_error","scale"],"properties":{"id":{"type":"string"},"name":{"type":"string"},"tracks":{"type":"array","items":{"type":"string"}},"reconstruction":{"type":"array","items":{"type":"object","required":["frame","matrix","average_error","matrix_is_camera_to_world"],"properties":{"frame":{"type":"number"},"matrix":{"type":"array","items":{"type":"number"},"minItems":16,"maxItems":16},"average_error":{"type":"number"},"matrix_is_camera_to_world":{"type":"boolean"}},"additionalProperties":false}},"reconstruction_is_valid":{"type":"boolean"},"reconstruction_average_error":{"type":"number"},"scale":{"type":"number"}},"additionalProperties":false});

    defs["color_management"] = json!({"type":"object","required":["display_device","view_transform","look","exposure","gamma","curve"],"properties":{"display_device":{"type":"string"},"view_transform":{"enum":["standard","ag_x","filmic","raw","false_color"]},"look":{"type":"string"},"exposure":{"type":"number"},"gamma":{"type":"number"},"curve":{"anyOf":[{"type":"null"},{"$ref":"#/$defs/curve_mapping"}]}},"additionalProperties":false});
    defs["curve_mapping"] = json!({"type":"object","required":["tone","extend_extrapolate","use_clip","clip","black","white","curves"],"properties":{"tone":{"enum":["standard","film_like"]},"extend_extrapolate":{"type":"boolean"},"use_clip":{"type":"boolean"},"clip":{"type":"array","items":{"type":"number"},"minItems":4,"maxItems":4},"black":{"type":"array","items":{"type":"number"},"minItems":3,"maxItems":3},"white":{"type":"array","items":{"type":"number"},"minItems":3,"maxItems":3},"curves":{"type":"array","items":{"type":"object","required":["points"],"properties":{"points":{"type":"array","minItems":2,"items":{"type":"object","required":["location","handle"],"properties":{"location":{"$ref":"#/$defs/vec2"},"handle":{"enum":["auto","auto_clamped","vector"]}},"additionalProperties":false}}},"additionalProperties":false},"minItems":4,"maxItems":4}},"additionalProperties":false});
    defs["sequencer"] = json!({"type":"object","required":["channels","strips"],"properties":{"channels":{"type":"integer","minimum":0,"maximum":u32::MAX},"strips":{"type":"array","items":{"type":"object","required":["id","name","type","channel","frame_start","frame_offset_start","frame_offset_end","length","blend_type","opacity","mute","retiming_keys","modifiers","sound_volume","sound_pan","sound_pitch","source","color","text","transition","inputs","effect"],"properties":{"id":{"type":"string"},"name":{"type":"string"},"type":{"enum":["image","image_sequence","movie","sound","scene","color","text","meta","transition","effect"]},"channel":{"type":"integer","minimum":0,"maximum":u32::MAX},"frame_start":{"type":"number"},"frame_offset_start":{"type":"number"},"frame_offset_end":{"type":"number"},"length":{"type":"number"},"blend_type":{"enum":["replace","alpha_over","add","subtract","multiply"]},"opacity":{"type":"number","minimum":0,"maximum":1},"mute":{"type":"boolean"},"retiming_keys":{"type":"array","items":{"type":"object","required":["frame","source_frame"],"properties":{"frame":{"type":"number"},"source_frame":{"type":"number"}},"additionalProperties":false}},"modifiers":{"type":"array","items":{"type":"object","required":["brightness","contrast","color_balance","curves"],"properties":{"brightness":{"type":"number"},"contrast":{"type":"number"},"color_balance":{"$ref":"#/$defs/vec3"},"curves":{"type":"array","items":{"$ref":"#/$defs/vec2"}}},"additionalProperties":false}},"sound_volume":{"type":"number","minimum":0},"sound_pan":{"type":"number","minimum":-1,"maximum":1},"sound_pitch":{"type":"number","exclusiveMinimum":0},"source":{"type":["string","null"]},"color":{"type":"array","items":{"type":"number"},"minItems":4,"maxItems":4},"text":{"type":["string","null"]},"transition":{"enum":["cross","gamma_cross","wipe",null]},"inputs":{"type":"array","items":{"type":"string"}},"effect":{"enum":["add","subtract","multiply","alpha_over","transform","speed","glow","gaussian_blur",null]}},"additionalProperties":false}}},"additionalProperties":false});
    defs["sequencer"]["properties"]["strips"]["items"]["properties"]["channel"]["minimum"] =
        json!(1);
    let scene = &mut defs["scene"];
    scene["required"] = json!([
        "name",
        "root_collection",
        "view_layers",
        "frame_current",
        "frame_start",
        "frame_end",
        "fps",
        "fps_base",
        "camera",
        "world",
        "active_clip",
        "unit",
        "render",
        "use_compositing",
        "compositor",
        "color_management",
        "markers",
        "rigid_body_world",
        "sequencer"
    ]);
    scene["properties"]["active_clip"] = json!({"oneOf":[{"$ref":"#/$defs/id"},{"type":"null"}]});
    scene["properties"]["use_compositing"] = json!({"type":"boolean"});
    scene["properties"]["compositor"] = json!({"oneOf":[{"$ref":"#/$defs/id"},{"type":"null"}]});
    scene["properties"]["color_management"] = json!({"$ref":"#/$defs/color_management"});
    scene["properties"]["sequencer"] = json!({"$ref":"#/$defs/sequencer"});
    let render = &mut defs["render_settings"];
    render["required"] = json!([
        "resolution_x",
        "resolution_y",
        "resolution_percentage",
        "samples",
        "seed",
        "max_bounces",
        "film_transparent",
        "engine",
        "use_sequencer",
        "audio_codec",
        "passes",
        "motion_blur",
        "shutter",
        "motion_blur_samples",
        "dicing_rate",
        "preview_dicing_rate",
        "max_subdivisions",
        "offscreen_dicing_scale"
    ]);
    render["properties"]["motion_blur"] = json!({"type":"boolean"});
    render["properties"]["shutter"] = json!({"type":"number"});
    render["properties"]["motion_blur_samples"] =
        json!({"type":"integer","minimum":0,"maximum":u32::MAX});
    render["properties"]["dicing_rate"] = json!({"type":"number","minimum":0.1,"maximum":1000.0});
    render["properties"]["preview_dicing_rate"] =
        json!({"type":"number","minimum":0.1,"maximum":1000.0});
    render["properties"]["max_subdivisions"] = json!({"type":"integer","minimum":0,"maximum":16});
    render["properties"]["offscreen_dicing_scale"] =
        json!({"type":"number","minimum":1.0,"maximum":f32::MAX});
    render["properties"]["dicing_camera"] = json!({"oneOf":[id_schema(),{"type":"null"}]});

    defs["node"]["properties"]["kind"]["enum"] =
        json!(["empty", "mesh", "camera", "light", "armature"]);
    let material = &mut defs["material"];
    material["properties"]["use_transparent_shadow"] = json!({"type":"boolean"});
    material["required"] = json!([
        "name",
        "base_color",
        "alpha_mode",
        "alpha_threshold",
        "metallic",
        "roughness",
        "emission_color",
        "emission_strength",
        "transmission",
        "ior",
        "double_sided",
        "node_tree",
        "base_color_texture",
        "roughness_texture",
        "metallic_texture",
        "normal_texture",
        "displacement_method",
        "displacement_scale",
        "displacement_midlevel",
        "volume_density",
        "volume_color",
        "volume_anisotropy"
    ]);
    material["properties"]["alpha_mode"] = json!({"enum":["opaque","blend","clip"]});
    material["properties"]["alpha_threshold"] = json!({"type":"number","minimum":0,"maximum":1});
    material["properties"]["displacement_method"] = json!({"enum":["bump","displacement"]});
    material["properties"]["displacement_scale"] = json!({"type":"number"});
    material["properties"]["displacement_midlevel"] =
        json!({"type":"number","minimum":0,"maximum":1});
    material["properties"]["volume_density"] = json!({"type":"number","minimum":0});
    material["properties"]["volume_color"] =
        json!({"type":"array","items":{"type":"number","minimum":0},"minItems":3,"maxItems":3});
    material["properties"]["volume_anisotropy"] =
        json!({"type":"number","minimum":-0.999,"maximum":0.999});
    complete_data_block_schemas(defs);
}

fn complete_data_block_schemas(defs: &mut Value) {
    defs["curve_data"] = json!({"type":"object","required":["resolution_u","splines","dimensions","bevel_depth","bevel_resolution","extrude","taper","fill_mode","twist_mode","use_path","path_duration","eval_time","eval_time_fcurves"],"properties":{"resolution_u":{"type":"integer","minimum":0,"maximum":u32::MAX},"splines":{"type":"array","items":{"type":"object","required":["type","points","order","cyclic","resolution","use_endpoint"],"properties":{"type":{"enum":["bezier","poly","nurbs"]},"points":{"type":"array","items":{"type":"object","required":["co","handle_left","handle_right","handle_type","weight","radius","tilt"],"properties":{"co":{"$ref":"#/$defs/vec3"},"handle_left":{"$ref":"#/$defs/vec3"},"handle_right":{"$ref":"#/$defs/vec3"},"handle_type":{"enum":["auto","vector","aligned","free"]},"weight":{"type":"number"},"radius":{"type":"number"},"tilt":{"type":"number"}},"additionalProperties":false}},"order":{"type":"integer","minimum":0,"maximum":u32::MAX},"cyclic":{"type":"boolean"},"resolution":{"type":"integer","minimum":0,"maximum":u32::MAX},"use_endpoint":{"type":"boolean"}},"additionalProperties":false}},"dimensions":{"enum":["two_d","three_d"]},"bevel_depth":{"type":"number"},"bevel_resolution":{"type":"integer","minimum":0,"maximum":u32::MAX},"extrude":{"type":"number"},"taper":{"oneOf":[{"$ref":"#/$defs/id"},{"type":"null"}]},"fill_mode":{"enum":["none","front","back","both"]},"twist_mode":{"type":"string"},"use_path":{"type":"boolean"},"path_duration":{"type":"integer","minimum":0,"maximum":u32::MAX},"eval_time":{"type":"number"},"eval_time_fcurves":{"type":"array","items":{"$ref":"#/$defs/fcurve"}}},"additionalProperties":false});
    defs["surface_data"] = json!({"type":"object","required":["points","order_u","order_v","resolution","cyclic_u","cyclic_v","use_endpoint_u","use_endpoint_v"],"properties":{"points":{"type":"array","items":{"type":"array","items":{"type":"object","required":["co","weight"],"properties":{"co":{"$ref":"#/$defs/vec3"},"weight":{"type":"number"}},"additionalProperties":false}}},"order_u":{"type":"integer","minimum":0,"maximum":u32::MAX},"order_v":{"type":"integer","minimum":0,"maximum":u32::MAX},"resolution":{"type":"array","items":{"type":"integer","minimum":0,"maximum":u32::MAX},"minItems":2,"maxItems":2},"cyclic_u":{"type":"boolean"},"cyclic_v":{"type":"boolean"},"use_endpoint_u":{"type":"boolean"},"use_endpoint_v":{"type":"boolean"}},"additionalProperties":false});
    defs["text_data"] = json!({"type":"object","required":["body","font","font_name","size","align_x","align_y","extrude","bevel_depth","character_spacing","word_spacing","line_spacing","shear","offset_x","offset_y","small_caps_scale"],"properties":{"body":{"type":"string"},"font":{"type":"string"},"font_name":{"type":"string"},"size":{"type":"number"},"align_x":{"type":"string"},"align_y":{"type":"string"},"extrude":{"type":"number"},"bevel_depth":{"type":"number"},"character_spacing":{"type":"number"},"word_spacing":{"type":"number"},"line_spacing":{"type":"number"},"shear":{"type":"number"},"offset_x":{"type":"number"},"offset_y":{"type":"number"},"small_caps_scale":{"type":"number"}},"additionalProperties":false});
    defs["hair_curves_data"] = json!({"type":"object","required":["curves"],"properties":{"curves":{"type":"array","items":{"type":"object","required":["points","radius"],"properties":{"points":{"type":"array","items":{"$ref":"#/$defs/vec3"}},"radius":{"type":"number"}},"additionalProperties":false}}},"additionalProperties":false});
    defs["metaball_data"] = json!({"type":"object","required":["elements","resolution","render_resolution","threshold"],"properties":{"elements":{"type":"array","items":{"type":"object","required":["type","co","radius","stiffness","size","rotation","negative"],"properties":{"type":{"enum":["ball","capsule","plane","ellipsoid","cube"]},"co":{"$ref":"#/$defs/vec3"},"radius":{"type":"number"},"stiffness":{"type":"number"},"size":{"$ref":"#/$defs/vec3"},"rotation":{"$ref":"#/$defs/vec3"},"negative":{"type":"boolean"}},"additionalProperties":false}},"resolution":{"type":"number"},"render_resolution":{"type":"number"},"threshold":{"type":"number"}},"additionalProperties":false});
    defs["lattice_data"] = json!({"type":"object","required":["points_u","points_v","points_w","interpolation","points","domain"],"properties":{"points_u":{"type":"integer","minimum":0},"points_v":{"type":"integer","minimum":0},"points_w":{"type":"integer","minimum":0},"interpolation":{"enum":["linear","cardinal","catmull_rom","bspline"]},"points":{"type":"array","items":{"$ref":"#/$defs/vec3"}},"domain":{"type":"object","required":["min","max"],"properties":{"min":{"$ref":"#/$defs/vec3"},"max":{"$ref":"#/$defs/vec3"}},"additionalProperties":false}},"additionalProperties":false});
    defs["point_cloud_data"] = json!({"type":"object","required":["points","attributes","next_id"],"properties":{"points":{"type":"array","items":{"type":"object","required":["id","position","radius"],"properties":{"id":{"type":"integer","minimum":0,"maximum":u32::MAX},"position":{"$ref":"#/$defs/vec3"},"radius":{"type":"number"}},"additionalProperties":false}},"attributes":{"type":"object","additionalProperties":{"type":"object","required":["domain","values"],"properties":{"domain":{"type":"string"},"values":{"type":"object"}},"additionalProperties":false}},"next_id":{"type":"integer","minimum":0,"maximum":u32::MAX}},"additionalProperties":false});
    defs["volume_data"] = json!({"type":"object","required":["grids","source"],"properties":{"grids":{"type":"array","items":{"type":"object","required":["dims","voxel_size","origin","values","blob_f32","content_ref"],"properties":{"dims":{"type":"array","items":{"type":"integer","minimum":0,"maximum":u32::MAX},"minItems":3,"maxItems":3},"voxel_size":{"type":"number"},"origin":{"$ref":"#/$defs/vec3"},"values":{"oneOf":[{"type":"array","items":{"type":"number"}},{"type":"null"}]},"blob_f32":{"oneOf":[{"type":"array","items":{"type":"integer","minimum":0,"maximum":255}},{"type":"null"}]},"content_ref":{"type":["string","null"]}},"additionalProperties":false}},"source":{"oneOf":[{"type":"object","required":["kind","metadata"],"properties":{"kind":{"const":"generated"},"metadata":{"type":"object","required":["algorithm"],"properties":{"algorithm":{"type":"string"}},"additionalProperties":false}},"additionalProperties":false},{"type":"object","required":["kind","metadata"],"properties":{"kind":{"const":"file"},"metadata":{"type":"object","required":["format","content_ref","grid_names","bounds_min","bounds_max"],"properties":{"format":{"type":"string"},"content_ref":{"type":["string","null"]},"grid_names":{"type":"array","items":{"type":"string"}},"bounds_min":{"oneOf":[{"$ref":"#/$defs/vec3"},{"type":"null"}]},"bounds_max":{"oneOf":[{"$ref":"#/$defs/vec3"},{"type":"null"}]}},"additionalProperties":false}},"additionalProperties":false}]}},"additionalProperties":false});
    defs["volume_data"]["properties"]["grids"]["items"]["properties"]["index_to_local"] = json!({"oneOf":[{"type":"array","items":{"type":"number"},"minItems":16,"maxItems":16},{"type":"null"}],"description":"Optional column-major 4x4 affine matrix mapping grid index coordinates to volume-local space; omitted uses origin and voxel_size."});
    for axis in ["points_u", "points_v", "points_w"] {
        defs["lattice_data"]["properties"][axis]["minimum"] = json!(2);
        defs["lattice_data"]["properties"][axis]["maximum"] = json!(256);
    }
    defs["point_cloud_data"]["properties"]["points"]["items"]["properties"]["radius"]["minimum"] =
        json!(0);
    defs["point_cloud_data"]["properties"]["attributes"]["additionalProperties"]["properties"]["domain"]
        ["const"] = json!("points");
    defs["volume_data"]["properties"]["grids"]["items"]["properties"]["dims"]["items"]["minimum"] =
        json!(1);
    for (field, definition) in [
        ("curve", "curve_data"),
        ("surface", "surface_data"),
        ("text", "text_data"),
        ("hair_curves", "hair_curves_data"),
        ("metaball", "metaball_data"),
        ("lattice", "lattice_data"),
        ("pointcloud", "point_cloud_data"),
        ("volume", "volume_data"),
    ] {
        defs["data_block"]["properties"][field] =
            json!({"oneOf":[{"$ref":format!("#/$defs/{definition}")},{"type":"null"}]});
    }
}

/// A texture reference: an image, an optional UV layer name (`null` is the render layer), and the
/// sampler interpolation.
fn texture_ref_schema() -> Value {
    json!({"type":"object","required":["image"],"properties":{"image":id_schema(),"uv_layer":{"type":["string","null"],"minLength":1,"default":null},"interpolation":{"enum":["linear","closest"],"default":"linear"},"extension":{"enum":["repeat","extend","clip","mirror"],"default":"repeat"}},"additionalProperties":false})
}

fn nullable_texture_ref_schema() -> Value {
    json!({"oneOf":[texture_ref_schema(),{"type":"null"}]})
}
/// A UV layer name: a non-empty string.
fn layer_name_schema() -> Value {
    json!({"type":"string","minLength":1})
}
fn avatar_settings_schema() -> Value {
    let id = id_schema();
    let nullable_id = json!({"oneOf":[id.clone(),{"type":"null"}]});
    let texture_ref = texture_ref_schema();
    let node_ref = json!({"type":"object","required":["node"],"properties":{"node":id.clone(),"bone":nullable_id.clone()},"additionalProperties":false});
    let expression_override = json!({"enum":["none","block","blend"]});
    let expressions = json!({
        "type":"object",
        "propertyNames":{"type":"string","minLength":1},
        "additionalProperties":{
            "type":"object",
            "properties":{
                "is_binary":{"type":"boolean","default":false},
                "override_blink":expression_override,
                "override_mouth":expression_override,
                "override_look_at":expression_override,
                "morph_target_binds":{"type":"array","items":{"type":"object","required":["node","shape_key","weight"],"properties":{"node":id.clone(),"shape_key":id.clone(),"weight":{"type":"number","minimum":0,"maximum":1}},"additionalProperties":false}},
                "material_color_binds":{"type":"array","items":{"type":"object","required":["material","color_type","target_value"],"properties":{"material":id.clone(),"color_type":{"enum":["color","emissionColor","shadeColor","matcapColor","rimColor","outlineColor"]},"target_value":vec_schema(4)},"additionalProperties":false}},
                "texture_transform_binds":{"type":"array","items":{"type":"object","required":["material","scale","offset"],"properties":{"material":id.clone(),"scale":vec_schema(2),"offset":vec_schema(2)},"additionalProperties":false}}
            },
            "additionalProperties":false
        }
    });
    let first_person = json!({"type":"object","properties":{"mesh_annotations":{"type":"array","items":{"type":"object","required":["node","annotation_type"],"properties":{"node":id.clone(),"annotation_type":{"enum":["auto","both","third_person_only","first_person_only"]}},"additionalProperties":false}}},"additionalProperties":false});
    let range = json!({"type":"object","required":["input_max_value","output_scale"],"properties":{"input_max_value":{"type":"number","minimum":0,"maximum":180},"output_scale":{"type":"number","minimum":0}},"additionalProperties":false});
    let mut look_at = json!({"type":"object","required":["look_at_type","offset_from_head_bone","range_map_horizontal_inner","range_map_horizontal_outer","range_map_vertical_down","range_map_vertical_up"],"properties":{"look_at_type":{"enum":["bone","expression"]},"offset_from_head_bone":vec_schema(3),"range_map_horizontal_inner":range,"range_map_horizontal_outer":range,"range_map_vertical_down":range,"range_map_vertical_up":range},"additionalProperties":false});
    look_at["allOf"] = json!([
        {"if":{"required":["look_at_type"],"properties":{"look_at_type":{"const":"expression"}}},"then":{"properties":{
            "range_map_horizontal_inner":{"properties":{"output_scale":{"maximum":1}}},
            "range_map_horizontal_outer":{"properties":{"output_scale":{"maximum":1}}},
            "range_map_vertical_down":{"properties":{"output_scale":{"maximum":1}}},
            "range_map_vertical_up":{"properties":{"output_scale":{"maximum":1}}}
        }}},
        {"if":{"required":["look_at_type"],"properties":{"look_at_type":{"const":"bone"}}},"then":{"properties":{
            "range_map_horizontal_inner":{"properties":{"output_scale":{"maximum":180}}},
            "range_map_horizontal_outer":{"properties":{"output_scale":{"maximum":180}}},
            "range_map_vertical_down":{"properties":{"output_scale":{"maximum":180}}},
            "range_map_vertical_up":{"properties":{"output_scale":{"maximum":180}}}
        }}}
    ]);
    let collider_shape = json!({"oneOf":[
        {"type":"object","required":["type","offset","radius"],"properties":{"type":{"const":"sphere"},"offset":vec_schema(3),"radius":{"type":"number","minimum":0}},"additionalProperties":false},
        {"type":"object","required":["type","offset","tail","radius"],"properties":{"type":{"const":"capsule"},"offset":vec_schema(3),"tail":vec_schema(3),"radius":{"type":"number","minimum":0}},"additionalProperties":false}
    ]});
    let spring = json!({"type":"object","properties":{
        "colliders":{"type":"array","items":{"type":"object","required":["node","shape"],"properties":{"node":node_ref,"shape":collider_shape},"additionalProperties":false}},
        "collider_groups":{"type":"array","items":{"type":"object","required":["name","colliders"],"properties":{"name":{"type":"string","minLength":1},"colliders":{"type":"array","items":{"type":"integer","minimum":0},"minItems":1,"uniqueItems":true}},"additionalProperties":false}},
        "springs":{"type":"array","items":{"type":"object","required":["joints"],"properties":{"name":{"type":["string","null"]},"center":{"oneOf":[node_ref,{"type":"null"}]},"joints":{"type":"array","minItems":1,"items":{"type":"object","required":["node"],"properties":{"node":node_ref,"hit_radius":{"type":"number","minimum":0,"default":0},"stiffness":{"type":"number","minimum":0,"default":1},"gravity_power":{"type":"number","minimum":0,"default":0},"gravity_dir":{"type":"array","items":{"type":"number"},"minItems":3,"maxItems":3,"default":[0,0,-1]},"drag_force":{"type":"number","minimum":0,"maximum":1,"default":0.5}},"additionalProperties":false}},"collider_groups":{"type":"array","items":{"type":"integer","minimum":0},"uniqueItems":true}},"additionalProperties":false}}
    },"additionalProperties":false});
    let texture = json!({"oneOf":[texture_ref.clone(),{"type":"null"}]});
    let mtoon = json!({"type":"object","properties":{
        "transparent_with_z_write":{"type":"boolean","default":false},
        "render_queue_offset_number":{"type":"integer","minimum":-9,"maximum":9,"default":0},
        "shade_color_factor":{"type":"array","items":{"type":"number","minimum":0,"maximum":1},"minItems":3,"maxItems":3,"default":[1,1,1]},
        "shade_multiply_texture":texture,
        "shading_shift_factor":{"type":"number","default":0},
        "shading_shift_texture":{"oneOf":[{"type":"object","required":["texture","scale"],"properties":{"texture":texture_ref,"scale":{"type":"number","default":1}},"additionalProperties":false},{"type":"null"}]},
        "shading_toony_factor":{"type":"number","minimum":0,"maximum":1,"default":0.9},
        "gi_equalization_factor":{"type":"number","minimum":0,"maximum":1,"default":0.9},
        "matcap_factor":{"type":"array","items":{"type":"number","minimum":0,"maximum":1},"minItems":3,"maxItems":3,"default":[1,1,1]},
        "matcap_texture":texture,
        "parametric_rim_color_factor":{"type":"array","items":{"type":"number","minimum":0,"maximum":1},"minItems":3,"maxItems":3,"default":[0,0,0]},
        "rim_multiply_texture":texture,
        "rim_lighting_mix_factor":{"type":"number","minimum":0,"maximum":1,"default":1},
        "parametric_rim_fresnel_power_factor":{"type":"number","minimum":0,"default":5},
        "parametric_rim_lift_factor":{"type":"number","default":0},
        "outline_width_mode":{"enum":["none","world_coordinates","screen_coordinates"],"default":"none"},
        "outline_width_factor":{"type":"number","minimum":0,"default":0},
        "outline_width_multiply_texture":texture,
        "outline_color_factor":{"type":"array","items":{"type":"number","minimum":0,"maximum":1},"minItems":3,"maxItems":3,"default":[0,0,0]},
        "outline_lighting_mix_factor":{"type":"number","minimum":0,"maximum":1,"default":1},
        "uv_animation_mask_texture":texture,
        "uv_animation_scroll_x_speed_factor":{"type":"number","default":0},
        "uv_animation_scroll_y_speed_factor":{"type":"number","default":0},
        "uv_animation_rotation_speed_factor":{"type":"number","default":0},
        "texture_transforms":{"type":"object","propertyNames":{"enum":["shadeMultiplyTexture","shadingShiftTexture","matcapTexture","rimMultiplyTexture","outlineWidthMultiplyTexture","uvAnimationMaskTexture"]},"additionalProperties":{"type":"object","properties":{"offset":vec_schema(2),"rotation":{"type":"number","default":0},"scale":vec_schema(2)},"additionalProperties":false}}
    },"additionalProperties":false});
    json!({
        "type":"object","required":["armature","humanoid","meta"],
        "properties":{
            "armature":id,
            "humanoid":{"type":"object","propertyNames":{"enum":["hips","spine","chest","upperChest","neck","head","leftEye","rightEye","jaw","leftUpperLeg","leftLowerLeg","leftFoot","leftToes","rightUpperLeg","rightLowerLeg","rightFoot","rightToes","leftShoulder","leftUpperArm","leftLowerArm","leftHand","rightShoulder","rightUpperArm","rightLowerArm","rightHand","leftThumbMetacarpal","leftThumbProximal","leftThumbDistal","leftIndexProximal","leftIndexIntermediate","leftIndexDistal","leftMiddleProximal","leftMiddleIntermediate","leftMiddleDistal","leftRingProximal","leftRingIntermediate","leftRingDistal","leftLittleProximal","leftLittleIntermediate","leftLittleDistal","rightThumbMetacarpal","rightThumbProximal","rightThumbDistal","rightIndexProximal","rightIndexIntermediate","rightIndexDistal","rightMiddleProximal","rightMiddleIntermediate","rightMiddleDistal","rightRingProximal","rightRingIntermediate","rightRingDistal","rightLittleProximal","rightLittleIntermediate","rightLittleDistal"]},"required":["hips","spine","head","leftUpperLeg","leftLowerLeg","leftFoot","rightUpperLeg","rightLowerLeg","rightFoot","leftUpperArm","leftLowerArm","leftHand","rightUpperArm","rightLowerArm","rightHand"],"additionalProperties":id},
            "meta":{"type":"object","required":["name","authors","license_url"],"properties":{
                "name":{"type":"string","minLength":1},"version":{"type":["string","null"]},"authors":{"type":"array","minItems":1,"items":{"type":"string","minLength":1}},
                "copyright_information":{"type":["string","null"]},"contact_information":{"type":["string","null"]},"references":{"type":"array","items":{"type":"string"}},
                "third_party_licenses":{"type":["string","null"]},"thumbnail":nullable_id,"license_url":{"type":"string","minLength":1},
                "avatar_permission":{"enum":["only_author","only_separately_licensed_person","everyone"],"default":"only_author"},
                "allow_excessively_violent_usage":{"type":"boolean","default":false},"allow_excessively_sexual_usage":{"type":"boolean","default":false},
                "commercial_usage":{"enum":["personal_non_profit","personal_profit","corporation"],"default":"personal_non_profit"},
                "allow_political_or_religious_usage":{"type":"boolean","default":false},"allow_antisocial_or_hate_usage":{"type":"boolean","default":false},
                "credit_notation":{"enum":["required","unnecessary"],"default":"required"},"allow_redistribution":{"type":"boolean","default":false},
                "modification":{"enum":["prohibited","allow_modification","allow_modification_redistribution"],"default":"prohibited"},
                "other_license_url":{"type":["string","null"]},
                "legacy":{"oneOf":[{"type":"object","properties":{
                    "title":{"type":["string","null"]},"author":{"type":["string","null"]},"reference":{"type":["string","null"]},
                    "allowed_user":{"oneOf":[{"enum":["only_author","explicitly_licensed_person","everyone"]},{"type":"null"}]},
                    "violent_usage":{"oneOf":[{"enum":["disallow","allow"]},{"type":"null"}]},"sexual_usage":{"oneOf":[{"enum":["disallow","allow"]},{"type":"null"}]},"commercial_usage":{"oneOf":[{"enum":["disallow","allow"]},{"type":"null"}]},
                    "other_permission_url":{"type":["string","null"]},"license_name":{"oneOf":[{"enum":["redistribution_prohibited","cc0","cc_by","cc_by_nc","cc_by_sa","cc_by_nc_sa","cc_by_nd","cc_by_nc_nd","other"]},{"type":"null"}]},
                    "other_license_url":{"type":["string","null"]}
                },"allOf":[{"if":{"required":["license_name"],"properties":{"license_name":{"const":"other"}}},"then":{"required":["other_license_url"],"properties":{"other_license_url":{"type":"string","minLength":1}}}}],"additionalProperties":false},{"type":"null"}]}
            },"additionalProperties":false},
            "expressions":expressions,
            "first_person":{"oneOf":[first_person,{"type":"null"}]},
            "look_at":{"oneOf":[look_at,{"type":"null"}]},
            "spring_bone":{"oneOf":[spring,{"type":"null"}]},
            "mtoon":{"type":"object","propertyNames":{"$ref":"#/$defs/id"},"additionalProperties":mtoon}
        },
        "additionalProperties":false
    })
}

fn avatar_update_set_schema() -> Value {
    let settings = avatar_settings_schema();
    let mut properties = settings["properties"].clone();
    for field in ["expressions", "mtoon"] {
        let current = properties[field].clone();
        properties[field] = json!({"oneOf":[current,{"type":"null"}]});
    }
    json!({"type":"object","minProperties":1,"properties":properties,"additionalProperties":false})
}

fn registry_schema(definition: &str) -> Value {
    json!({"type":"object","propertyNames":{"$ref":"#/$defs/id"},"additionalProperties":{"$ref":format!("#/$defs/{definition}")}})
}

fn operations_schema() -> Value {
    let mut defs = Map::new();
    defs.insert("id".to_owned(), id_schema());
    defs.insert("keyframe".to_owned(), keyframe_schema());
    defs.insert("fcurve".to_owned(), fcurve_schema());
    for name in ops::OP_NAMES {
        defs.insert(def_key(name), operation_schema(name));
    }
    let variants = ops::OP_NAMES
        .iter()
        .map(|name| json!({"$ref":format!("#/$defs/{}",def_key(name))}))
        .collect::<Vec<_>>();
    let mut root = base("Potter operation batch");
    root["required"] = json!(["schema_version", "base_revision", "operations"]);
    root["properties"] = json!({
        "schema_version":{"const":1},"base_revision":{"type":"integer","minimum":0,"maximum":9_007_199_254_740_991_u64},
        "operations":{"type":"array","maxItems":10000,"items":{"oneOf":variants}},
        "evaluation":{"type":"object","properties":{"scene_id":{"$ref":"#/$defs/id"},"view_layer":{"$ref":"#/$defs/id"},"frame":{"type":"number"},"camera":{"$ref":"#/$defs/id"},"mode":{"type":"string"}},"additionalProperties":false}
    });
    root["$defs"] = Value::Object(defs);
    root
}

fn operation_schema(name: &str) -> Value {
    let contract = operation_contract(name);
    let mut properties = Map::new();
    properties.insert("op".to_owned(), json!({"const":name}));
    for field in &contract.fields {
        let schema = field_schema(name, field);
        properties.insert((*field).to_owned(), schema);
    }
    let mut result = json!({"$schema":DRAFT,"title":format!("Operation {name}"),"type":"object","properties":properties,"required":contract.required,"additionalProperties":false,
        "$defs":{"id":id_schema(),"keyframe":keyframe_schema(),"fcurve":fcurve_schema()},"x-potter":{"target_type":contract.target_type,"scope":contract.scope,"reversible":contract.reversible,"defaults":contract.defaults}});
    if name == "node.delete" {
        result["allOf"] = json!([
            {"not":{"required":["recursive","reparent"],"properties":{"recursive":{"const":true}}}},
            {"if":{"required":["keep_world"]},"then":{"required":["reparent"]}}
        ]);
    }
    if name == "texture.create" {
        result["allOf"] = json!([{
            "if":{"properties":{"texture_type":{"const":"IMAGE"}},"required":["texture_type"]},
            "then":{"required":["image"],"properties":{"settings":image_texture_settings_schema()}},
            "else":{"not":{"required":["image"]}}
        }]);
    }
    if name == "collection.link" || name == "collection.unlink" {
        result["oneOf"] = json!([{"required":["object"],"not":{"required":["child"]}},{"required":["child"],"not":{"required":["object"]}}]);
    }
    if name == "node.join" {
        result["oneOf"] = json!([
            {"required":["target"],"not":{"required":["targets"]}},
            {"required":["targets"],"not":{"required":["target"]}}
        ]);
    }
    if name == "node.create" {
        let primitive_kinds = [
            "box",
            "uv_sphere",
            "sphere",
            "cylinder",
            "plane",
            "cone",
            "torus",
            "icosphere",
            "circle",
            "grid",
        ];
        let non_primitive_kinds = [
            "empty",
            "mesh",
            "camera",
            "light",
            "curve",
            "surface",
            "text",
            "metaball",
            "lattice",
            "pointcloud",
            "volume",
            "armature",
            "grease_pencil",
            "collection_instance",
            "group",
        ];
        let mut conditions = vec![json!({
            "if":{"properties":{"kind":{"enum":primitive_kinds}},"required":["kind"]},
            "then":{"required":["params"],"not":{"required":["data"]}}
        })];
        for kind in primitive_kinds.into_iter().chain(non_primitive_kinds) {
            conditions.push(json!({
                "if":{"properties":{"kind":{"const":kind}},"required":["kind"]},
                "then":{"properties":{"params":primitive_params_schema(kind)}}
            }));
        }
        result["allOf"] = Value::Array(conditions);
    }
    if matches!(name, "modifier.create" | "modifier.update") {
        result["allOf"] = parameter_conditions(ParameterFamily::Modifier, name);
    }
    if matches!(name, "constraint.create" | "constraint.update") {
        result["allOf"] = parameter_conditions(ParameterFamily::Constraint, name);
    }
    if name == "physics.rigid_body.create" {
        result["allOf"] = json!([{"if":{"properties":{"type":{"const":"active"}},"required":["type"]},"then":{"properties":{"mass":{"exclusiveMinimum":0}}}}]);
    }
    if [
        "mesh.transform_elements",
        "mesh.extrude",
        "mesh.inset",
        "mesh.bevel",
        "mesh.subdivide",
        "mesh.poke",
        "mesh.delete",
        "mesh.dissolve",
        "mesh.weld",
        "mesh.fill",
        "mesh.flip_normals",
        "mesh.set_attribute",
        "mesh.attribute_update",
        "mesh.loop_cut",
        "mesh.edge_slide",
        "mesh.vertex_slide",
        "mesh.spin",
        "mesh.screw",
        "mesh.merge",
        "mesh.rip",
        "mesh.shade_smooth",
        "mesh.shade_flat",
        "mesh.set_custom_normals",
        "mesh.bridge",
        "mesh.split",
        "mesh.knife",
    ]
    .contains(&name)
    {
        result["anyOf"] =
            json!([{"required":["elements"]},{"properties":{"target":{"required":["elements"]}}}]);
    }
    if name == "mesh.separate" {
        result["allOf"] = json!([{
            "if":{"properties":{"mode":{"const":"selection"}},"required":["mode"]},
            "then":{"anyOf":[{"required":["elements"]},{"properties":{"target":{"required":["elements"]}}}]}
        }]);
    }
    result
}
fn image_texture_settings_schema() -> Value {
    json!({
        "type":"object",
        "properties":{
            "use_interpolation":{"type":"boolean"},
            "filter_size":{"type":"number","minimum":0.1,"maximum":50.0},
            "use_flip_axis":{"type":"boolean"},
            "use_alpha":{"type":"boolean"},
            "use_calculate_alpha":{"type":"boolean"},
            "invert_alpha":{"type":"boolean"},
            "extension":{"enum":["EXTEND","CLIP","CLIP_CUBE","REPEAT","CHECKER"]},
            "repeat_x":{"type":"integer","minimum":1,"maximum":512},
            "repeat_y":{"type":"integer","minimum":1,"maximum":512},
            "use_mirror_x":{"type":"boolean"},
            "use_mirror_y":{"type":"boolean"},
            "use_checker_odd":{"type":"boolean"},
            "use_checker_even":{"type":"boolean"},
            "checker_distance":{"type":"number","minimum":0.0,"maximum":0.99},
            "crop_min_x":{"type":"number","minimum":-10.0,"maximum":10.0},
            "crop_min_y":{"type":"number","minimum":-10.0,"maximum":10.0},
            "crop_max_x":{"type":"number","minimum":-10.0,"maximum":10.0},
            "crop_max_y":{"type":"number","minimum":-10.0,"maximum":10.0},
            "intensity":{"type":"number","minimum":0.0,"maximum":2.0},
            "contrast":{"type":"number","minimum":0.0,"maximum":5.0},
            "saturation":{"type":"number","minimum":0.0,"maximum":2.0},
            "factor_red":{"type":"number","minimum":0.0,"maximum":2.0},
            "factor_green":{"type":"number","minimum":0.0,"maximum":2.0},
            "factor_blue":{"type":"number","minimum":0.0,"maximum":2.0},
            "use_clamp":{"type":"boolean"},
            "use_normal_map":{"type":"boolean"},
            "use_preview_alpha":{"type":"boolean"},
            "use_color_ramp":{"type":"boolean"},
            "color_ramp":{"type":"object"},
            "image_user":{"type":"object"}
        },
        "additionalProperties":false
    })
}

fn parameter_conditions(family: ParameterFamily, operation: &str) -> Value {
    let Some(type_specs) = params::type_specs(family) else {
        return Value::Array(Vec::new());
    };
    let is_update = operation.ends_with(".update");
    let conditions = type_specs
        .iter()
        .filter_map(|(kind, specification)| {
            let parameter_schema = params::params_schema(family, kind)?;
            if is_update {
                Some(json!({
                    "if":{
                        "properties":{"set":{"properties":{"type":{"const":kind}},"required":["type"]}},
                        "required":["set"]
                    },
                    "then":{"properties":{"set":{"properties":{"params":parameter_schema}}}}
                }))
            } else {
                let requires_params = specification
                    .get("required")
                    .and_then(Value::as_array)
                    .is_some_and(|required| !required.is_empty());
                let mut then = json!({"properties":{"params":parameter_schema}});
                if requires_params {
                    then["required"] = json!(["params"]);
                }
                Some(json!({
                    "if":{"properties":{"type":{"const":kind}},"required":["type"]},
                    "then":then
                }))
            }
        })
        .collect();
    Value::Array(conditions)
}

fn positive_parameter(default: f64) -> Value {
    json!({"type":"number","exclusiveMinimum":0,"default":default})
}

fn primitive_params_schema(kind: &str) -> Value {
    let properties = match kind {
        "box" | "plane" => json!({"size":positive_parameter(2.0)}),
        "grid" => json!({
            "size":positive_parameter(2.0),
            "x_subdivisions":{"type":"integer","minimum":1,"maximum":10_000_000,"default":10},
            "y_subdivisions":{"type":"integer","minimum":1,"maximum":10_000_000,"default":10}
        }),
        "uv_sphere" | "sphere" => json!({
            "segments":{"type":"integer","minimum":3,"maximum":100_000,"default":32},
            "ring_count":{"type":"integer","minimum":3,"maximum":100_000,"default":16},
            "radius":positive_parameter(1.0)
        }),
        "cylinder" => json!({
            "vertices":{"type":"integer","minimum":3,"maximum":10_000_000,"default":32},
            "radius":positive_parameter(1.0),
            "depth":positive_parameter(2.0),
            "end_fill_type":{"enum":["NOTHING","NGON","TRIFAN"],"default":"NGON"}
        }),
        "cone" => json!({
            "vertices":{"type":"integer","minimum":3,"maximum":10_000_000,"default":32},
            "radius1":{"type":"number","minimum":0,"default":1.0,"description":"At least one of radius1 and radius2 must be positive."},
            "radius2":{"type":"number","minimum":0,"default":0.0,"description":"At least one of radius1 and radius2 must be positive."},
            "depth":positive_parameter(2.0),
            "end_fill_type":{"enum":["NOTHING","NGON","TRIFAN"],"default":"NGON"}
        }),
        "torus" => json!({
            "major_radius":{"type":"number","exclusiveMinimum":0,"default":1.0},
            "minor_radius":{"type":"number","exclusiveMinimum":0,"default":0.25},
            "abso_major_rad":{"type":"number","exclusiveMinimum":0,"default":1.25},
            "abso_minor_rad":{"type":"number","minimum":0,"default":0.75},
            "major_segments":{"type":"integer","minimum":3,"maximum":256,"default":48},
            "minor_segments":{"type":"integer","minimum":3,"maximum":256,"default":12},
            "mode":{"enum":["MAJOR_MINOR","EXT_INT"],"default":"MAJOR_MINOR"}
        }),
        "icosphere" => json!({
            "subdivisions":{"type":"integer","minimum":1,"maximum":10,"default":2},
            "radius":positive_parameter(1.0)
        }),
        "circle" => json!({
            "vertices":{"type":"integer","minimum":3,"maximum":10_000_000,"default":32},
            "radius":positive_parameter(1.0),
            "fill_type":{"enum":["NOTHING","NGON","TRIFAN"],"default":"NOTHING"}
        }),
        _ => json!({}),
    };
    json!({"type":"object","properties":properties,"additionalProperties":false})
}

struct OperationContract {
    fields: Vec<&'static str>,
    required: Vec<&'static str>,
    target_type: &'static str,
    scope: &'static str,
    reversible: bool,
    defaults: Value,
}

fn operation_contract(name: &str) -> OperationContract {
    let (fields, required, target_type, scope, defaults) = match name {
        "library.link" => (
            vec!["id", "uri", "items"],
            vec!["op", "id", "uri", "items"],
            "library",
            "none",
            json!({}),
        ),
        "library.append" => (
            vec!["library_id", "uri", "items"],
            vec!["op", "items"],
            "library",
            "none",
            json!({}),
        ),
        "library.override" => (
            vec!["target", "id", "operations"],
            vec!["op", "target", "id", "operations"],
            "linked node",
            "none",
            json!({}),
        ),
        "library.register" => (
            vec![
                "id",
                "name",
                "kind",
                "uri",
                "resolved_path",
                "resource",
                "hash",
                "items",
            ],
            vec!["op", "id", "name", "kind", "uri", "resolved_path", "items"],
            "library",
            "none",
            json!({"kind":"blend","resource":null,"hash":"computed from resolved_path when omitted"}),
        ),
        "library.reload" => (vec!["id"], vec!["op", "id"], "library", "none", json!({})),
        "library.relocate" => (
            vec!["id", "uri", "resolved_path"],
            vec!["op", "id", "uri"],
            "library",
            "none",
            json!({}),
        ),
        "scene.create" => (
            vec!["id", "name", "root_collection", "view_layer", "active"],
            vec!["op", "id"],
            "scene",
            "none",
            json!({"name":"id","root_collection":"generated when omitted","view_layer":"view_main","active":false}),
        ),
        "scene.update" | "render.update" => (
            vec!["target", "set"],
            vec!["op", "target", "set"],
            "scene",
            "none",
            json!({}),
        ),
        "collection.create" => (
            vec!["id", "name", "parent"],
            vec!["op", "id"],
            "collection",
            "none",
            json!({"name":"id","parent":"active Scene root Collection"}),
        ),
        "collection.parent" => (
            vec!["target", "parent"],
            vec!["op", "target", "parent"],
            "collection",
            "none",
            json!({"parent":"collection ID or null to unparent"}),
        ),
        "collection.update" => (
            vec!["target", "set", "scene_id", "view_layer"],
            vec!["op", "target", "set"],
            "collection",
            "none",
            json!({"scene_id":"active Scene","view_layer":"first View Layer in the selected Scene"}),
        ),
        "collection.delete" => (
            vec!["target", "unlink"],
            vec!["op", "target"],
            "collection",
            "none",
            json!({"unlink":false}),
        ),
        "collection.link" | "collection.unlink" => (
            vec!["collection", "object", "child"],
            vec!["op", "collection"],
            "collection plus object or child",
            "none",
            json!({}),
        ),
        "node.create" => (
            vec![
                "id",
                "kind",
                "name",
                "tags",
                "parent",
                "parent_inverse",
                "transform",
                "params",
                "data",
                "material",
                "materials",
                "collection",
                "visible",
                "render_visible",
                "selectable",
                "track_axis",
                "up_axis",
            ],
            vec!["op", "id", "kind"],
            "object",
            "none",
            json!({"name":"id","collection":"active Scene root Collection","visible":true,"render_visible":true,"selectable":true,"params":"required for primitive kinds"}),
        ),
        "node.update" => (
            vec!["target", "scope", "set"],
            vec!["op", "target", "set"],
            "object",
            "shared|single_user for params",
            json!({}),
        ),
        "node.delete" => (
            vec![
                "target",
                "recursive",
                "reparent",
                "keep_world",
                "cascade_data",
                "unlink",
            ],
            vec!["op", "target"],
            "object",
            "none",
            json!({"recursive":false,"reparent":null,"keep_world":true,"cascade_data":false,"unlink":false}),
        ),
        "node.duplicate" => (
            vec!["target", "id", "mode", "recursive"],
            vec!["op", "target", "id"],
            "single object",
            "none",
            json!({"mode":"independent","recursive":false}),
        ),
        "node.parent" => (
            vec!["target", "parent", "keep_world"],
            vec!["op", "target", "parent"],
            "single object",
            "none",
            json!({"keep_world":true}),
        ),
        "node.join" => (
            vec!["target", "targets"],
            vec!["op"],
            "two or more mesh objects",
            "none",
            json!({"destination":"first selected mesh object"}),
        ),
        "data.make_single_user" => (
            vec!["target", "data_id"],
            vec!["op", "target"],
            "object",
            "none",
            json!({"data_id":"generated unique ID"}),
        ),
        "material.create" => (
            vec![
                "id",
                "name",
                "alpha_mode",
                "alpha_threshold",
                "base_color",
                "metallic",
                "roughness",
                "double_sided",
                "use_transparent_shadow",
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
            ],
            vec!["op", "id"],
            "material",
            "none",
            json!({"name":"id","base_color":[0.6,0.6,0.6,1.0],"metallic":0.0,"roughness":0.8,"double_sided":false,"emission_color":[0.0,0.0,0.0],"emission_strength":0.0,"transmission":0.0,"ior":1.45}),
        ),
        "material.update" => (
            vec!["target", "set"],
            vec!["op", "target", "set"],
            "material",
            "none",
            json!({}),
        ),
        "material.delete" => (
            vec!["target"],
            vec!["op", "target"],
            "material",
            "none",
            json!({}),
        ),
        "modifier.create" => (
            vec![
                "target",
                "id",
                "type",
                "name",
                "enabled",
                "show_viewport",
                "show_render",
                "params",
            ],
            vec!["op", "target", "id", "type"],
            "object",
            "object modifier stack",
            json!({"name":"type","enabled":true,"show_viewport":true,"show_render":true,"params":{}}),
        ),
        "modifier.update" => (
            vec!["target", "modifier_id", "set"],
            vec!["op", "target", "modifier_id", "set"],
            "object",
            "object modifier stack",
            json!({}),
        ),
        "modifier.delete" => (
            vec!["target", "modifier_id"],
            vec!["op", "target", "modifier_id"],
            "object",
            "object modifier stack",
            json!({}),
        ),
        "modifier.reorder" => (
            vec!["target", "modifier_id", "index"],
            vec!["op", "target", "modifier_id", "index"],
            "object",
            "object modifier stack",
            json!({}),
        ),
        "modifier.bind" => (
            vec!["target", "modifier_id"],
            vec!["op", "target", "modifier_id"],
            "mesh modifier stack",
            "stores bind-time deformation data in the modifier params",
            json!({}),
        ),
        "modifier.unbind" => (
            vec!["target", "modifier_id"],
            vec!["op", "target", "modifier_id"],
            "mesh modifier stack",
            "removes stored bind-time deformation data",
            json!({}),
        ),
        "modifier.apply" => (
            vec!["target", "modifier_id", "scope"],
            vec!["op", "target", "modifier_id"],
            "object",
            "shared|single_user for shared data",
            json!({}),
        ),
        "modifier.apply_as_shape_key" => (
            vec![
                "target",
                "modifier_id",
                "id",
                "name",
                "keep_modifier",
                "scope",
            ],
            vec!["op", "target", "modifier_id", "id"],
            "object",
            "topology-preserving modifier result stored as a shape key",
            json!({"keep_modifier":false}),
        ),
        "sculpt.stroke" => (
            vec![
                "target",
                "scope",
                "brush",
                "deform_type",
                "samples",
                "falloff",
                "symmetry",
                "delta",
                "view",
                "dyntopo",
            ],
            vec!["op", "target", "brush", "samples"],
            "mesh object and stroke samples",
            "shared|single_user for shared data",
            json!({"symmetry":[]}),
        ),
        "sculpt.dyntopo" => (
            vec![
                "target",
                "scope",
                "samples",
                "falloff",
                "symmetry",
                "edge_length",
            ],
            vec!["op", "target", "samples", "edge_length"],
            "mesh object and stroke samples",
            "shared|single_user for shared data",
            json!({"falloff":"smooth","symmetry":[]}),
        ),
        "mesh.transform_elements" => (
            vec![
                "target",
                "scope",
                "elements",
                "translation",
                "rotation",
                "scale",
                "pivot",
                "proportional",
            ],
            vec!["op", "target"],
            "mesh object and selected elements",
            "shared|single_user for shared data",
            json!({"pivot":[0,0,0]}),
        ),
        "mesh.extrude" => (
            vec![
                "target",
                "scope",
                "elements",
                "distance",
                "offset",
                "individual",
            ],
            vec!["op", "target"],
            "mesh object and selected elements",
            "shared|single_user for shared data",
            json!({}),
        ),
        "mesh.inset" => (
            vec!["target", "scope", "elements", "amount", "width", "distance"],
            vec!["op", "target"],
            "mesh object and selected elements",
            "shared|single_user for shared data",
            json!({"amount":0.1}),
        ),
        "mesh.edge_slide" | "mesh.vertex_slide" => (
            vec!["target", "scope", "elements", "factor"],
            vec!["op", "target"],
            "mesh object and selected edges or vertices",
            "shared|single_user for shared data",
            json!({"factor":0.5}),
        ),
        "mesh.spin" => (
            vec![
                "target", "scope", "elements", "axis", "angle", "center", "steps",
            ],
            vec!["op", "target", "axis", "angle"],
            "mesh object and selected elements",
            "shared|single_user for shared data",
            json!({"center":[0,0,0]}),
        ),
        "mesh.screw" => (
            vec![
                "target", "scope", "elements", "axis", "angle", "center", "distance", "steps",
            ],
            vec!["op", "target", "axis", "angle", "distance"],
            "mesh object and selected elements",
            "shared|single_user for shared data",
            json!({"center":[0,0,0]}),
        ),
        "mesh.merge" => (
            vec!["target", "scope", "elements", "mode", "cursor"],
            vec!["op", "target"],
            "mesh object and selected vertices",
            "shared|single_user for shared data",
            json!({"mode":"center"}),
        ),
        "mesh.rip" | "mesh.shade_smooth" | "mesh.shade_flat" => (
            vec!["target", "scope", "elements"],
            vec!["op", "target"],
            "mesh object and selected faces",
            "shared|single_user for shared data",
            json!({}),
        ),
        "mesh.auto_smooth" | "mesh.mark_sharp_by_angle" => (
            vec!["target", "scope", "elements", "angle"],
            vec!["op", "target"],
            "mesh object and edges",
            "shared|single_user for shared data",
            json!({"angle":std::f64::consts::FRAC_PI_4}),
        ),
        "mesh.set_custom_normals" => (
            vec!["target", "scope", "elements", "normals"],
            vec!["op", "target"],
            "mesh object and selected face corners",
            "shared|single_user for shared data",
            json!({}),
        ),
        "mesh.bevel" => (
            vec!["target", "scope", "elements", "width", "amount", "segments"],
            vec!["op", "target"],
            "mesh object and selected elements",
            "shared|single_user for shared data",
            json!({"width":0.1,"segments":1}),
        ),
        "mesh.subdivide" => (
            vec!["target", "scope", "elements", "cuts"],
            vec!["op", "target"],
            "mesh object and selected elements",
            "shared|single_user for shared data",
            json!({"cuts":1}),
        ),
        "mesh.triangulate" | "mesh.poke" | "mesh.delete" | "mesh.dissolve" | "mesh.fill"
        | "mesh.flip_normals" => (
            vec!["target", "scope", "elements"],
            vec!["op", "target"],
            "mesh object and selected elements",
            "shared|single_user for shared data",
            json!({}),
        ),
        "mesh.weld" => (
            vec!["target", "scope", "elements", "threshold", "distance"],
            vec!["op", "target"],
            "mesh object and selected elements",
            "shared|single_user for shared data",
            json!({"threshold":0.000_001}),
        ),
        "mesh.bisect" => (
            vec![
                "target",
                "scope",
                "elements",
                "plane",
                "point",
                "normal",
                "threshold",
                "clear_side",
            ],
            vec!["op", "target"],
            "mesh object",
            "shared|single_user for shared data",
            json!({"point":[0,0,0],"normal":[0,0,1],"threshold":0.000_000_000_1}),
        ),
        "mesh.mirror" => (
            vec![
                "target",
                "scope",
                "elements",
                "axis",
                "merge",
                "merge_threshold",
                "threshold",
                "origin",
            ],
            vec!["op", "target"],
            "mesh object",
            "shared|single_user for shared data",
            json!({"axis":"x","merge":false,"merge_threshold":0.000_001}),
        ),
        "mesh.set_attribute" => (
            vec![
                "target",
                "scope",
                "elements",
                "attribute",
                "name",
                "value",
                "material_index",
            ],
            vec!["op", "target"],
            "mesh object",
            "shared|single_user for shared data",
            json!({}),
        ),
        "mesh.attribute_create" => (
            vec!["target", "scope", "name", "domain", "type", "default"],
            vec!["op", "target", "name", "domain", "type"],
            "mesh object",
            "shared|single_user for shared data",
            json!({}),
        ),
        "mesh.attribute_update" => (
            vec!["target", "scope", "elements", "name", "value"],
            vec!["op", "target", "name", "value"],
            "mesh object and selected elements",
            "shared|single_user for shared data",
            json!({}),
        ),
        "mesh.attribute_delete" => (
            vec!["target", "scope", "elements", "name"],
            vec!["op", "target", "name"],
            "mesh object",
            "shared|single_user for shared data",
            json!({}),
        ),
        "mesh.separate" => (
            vec!["target", "scope", "mode", "elements"],
            vec!["op", "target", "mode"],
            "mesh object and separated face partitions",
            "shared|single_user for shared data",
            json!({"mode":"selection"}),
        ),
        "mesh.loop_cut" => (
            vec!["target", "scope", "elements", "cuts"],
            vec!["op", "target"],
            "mesh object and selected edges",
            "shared|single_user for shared data",
            json!({"cuts":1}),
        ),
        "mesh.bridge" | "mesh.split" => (
            vec!["target", "scope", "elements"],
            vec!["op", "target"],
            "mesh object and selected edges or faces",
            "shared|single_user for shared data",
            json!({}),
        ),
        "mesh.knife" => (
            vec![
                "target",
                "scope",
                "elements",
                "plane",
                "point",
                "normal",
                "threshold",
            ],
            vec!["op", "target"],
            "mesh object",
            "shared|single_user for shared data",
            json!({"point":[0,0,0],"normal":[0,0,1],"threshold":0.000_000_000_1}),
        ),
        "mesh.remesh" => (
            vec!["target", "scope", "voxel_size"],
            vec!["op", "target", "voxel_size"],
            "mesh object",
            "shared|single_user for shared data",
            json!({}),
        ),
        "mesh.symmetrize" => (
            vec![
                "target",
                "scope",
                "elements",
                "axis",
                "direction",
                "origin",
                "threshold",
            ],
            vec!["op", "target"],
            "mesh object",
            "shared|single_user for shared data",
            json!({"axis":"x","direction":"positive_to_negative","origin":[0,0,0],"threshold":0.000_001}),
        ),
        "uv.unwrap" => (
            vec!["target", "elements", "scope", "method", "uv_layer"],
            vec!["op", "target"],
            "mesh object and faces",
            "shared|single_user for shared data",
            json!({"method":"box"}),
        ),
        "uv.transform" => (
            vec![
                "target",
                "elements",
                "scope",
                "translation",
                "scale",
                "rotation",
                "uv_layer",
            ],
            vec!["op", "target"],
            "mesh object and faces",
            "shared|single_user for shared data",
            json!({"translation":[0,0],"scale":[1,1],"rotation":0}),
        ),
        "uv.pin" => (
            vec!["target", "elements", "scope", "pinned", "uv_layer"],
            vec!["op", "target"],
            "mesh object and faces",
            "shared|single_user for shared data",
            json!({"pinned":true}),
        ),
        "uv.pack" => (
            vec!["target", "elements", "scope", "margin", "uv_layer"],
            vec!["op", "target"],
            "mesh object and faces",
            "shared|single_user for shared data",
            json!({"margin":0.001}),
        ),
        "uv.layer_create" => (
            vec!["target", "scope", "name"],
            vec!["op", "target"],
            "mesh object",
            "shared|single_user for shared data",
            json!({}),
        ),
        "uv.layer_rename" => (
            vec!["target", "scope", "name", "to"],
            vec!["op", "target", "name", "to"],
            "mesh object",
            "shared|single_user for shared data",
            json!({}),
        ),
        "uv.layer_delete" | "uv.layer_set_active" | "uv.layer_set_render" => (
            vec!["target", "scope", "name"],
            vec!["op", "target", "name"],
            "mesh object",
            "shared|single_user for shared data",
            json!({}),
        ),
        "texture.create" => (
            vec!["id", "texture_type", "image", "settings"],
            vec!["op", "id", "texture_type"],
            "legacy texture",
            "none",
            json!({"image":"required for IMAGE; references a shared doc.images entry","settings":"Blender defaults"}),
        ),
        "texture.update" => (
            vec!["id", "image", "settings"],
            vec!["op", "id"],
            "legacy texture",
            "none",
            json!({"image":"unchanged unless supplied","settings":"merged over existing settings"}),
        ),
        "paint.texture" => (
            vec![
                "target", "scope", "image", "uv_layer", "samples", "falloff", "symmetry", "color",
                "blend",
            ],
            vec!["op", "target", "image", "samples", "color"],
            "mesh object and UV corners",
            "shared|single_user for shared data",
            json!({"blend":"mix"}),
        ),
        "bone.create" => (
            vec![
                "target",
                "id",
                "name",
                "head",
                "tail",
                "parent",
                "roll",
                "deform",
                "inherit_rotation",
                "use_connect",
                "custom_shape",
                "envelope_distance",
                "envelope_weight",
                "head_radius",
                "tail_radius",
            ],
            vec!["op", "target", "id", "name", "head", "tail"],
            "armature object",
            "none",
            json!({"deform":true,"inherit_rotation":true,"envelope_distance":0.25,"envelope_weight":1.0,"head_radius":0.1,"tail_radius":0.1}),
        ),
        "bone.update" => (
            vec!["target", "bone_id", "set"],
            vec!["op", "target", "bone_id", "set"],
            "armature object bone",
            "none",
            json!({}),
        ),
        "bone.delete" => (
            vec!["target", "bone_id"],
            vec!["op", "target", "bone_id"],
            "armature object bone",
            "none",
            json!({}),
        ),
        "bone_collection.create" => (
            vec!["target", "id", "name", "visible"],
            vec!["op", "target", "id", "name"],
            "armature object",
            "none",
            json!({"visible":true}),
        ),
        "bone_collection.update" => (
            vec!["target", "id", "set"],
            vec!["op", "target", "id", "set"],
            "armature object bone collection",
            "none",
            json!({}),
        ),
        "bone_collection.delete" => (
            vec!["target", "id"],
            vec!["op", "target", "id"],
            "armature object bone collection",
            "none",
            json!({}),
        ),
        "bone_collection.assign" | "bone_collection.unassign" => (
            vec!["target", "collection_id", "bone_ids"],
            vec!["op", "target", "collection_id", "bone_ids"],
            "armature object bone collection and bones",
            "none",
            json!({}),
        ),
        "constraint.create" => (
            vec![
                "target",
                "id",
                "type",
                "name",
                "constraint_target",
                "subtarget",
                "owner_bone",
                "influence",
                "enabled",
                "params",
            ],
            vec!["op", "target", "id", "type"],
            "object constraint or armature pose-bone constraint",
            "none",
            json!({"influence":1.0,"enabled":true,"owner_bone":null}),
        ),
        "constraint.update" => (
            vec!["target", "id", "set"],
            vec!["op", "target", "id", "set"],
            "object or pose-bone constraint",
            "none",
            json!({}),
        ),
        "constraint.delete" => (
            vec!["target", "id"],
            vec!["op", "target", "id"],
            "object or pose-bone constraint",
            "none",
            json!({}),
        ),
        "constraint.reorder" => (
            vec!["target", "id", "index"],
            vec!["op", "target", "id", "index"],
            "object or pose-bone constraint stack",
            "none",
            json!({}),
        ),
        "tracking.clip_create" => (
            vec![
                "id",
                "name",
                "source",
                "source_hash",
                "frame_start",
                "fps",
                "width",
                "height",
            ],
            vec!["op", "id", "name"],
            "movie clip",
            "none",
            json!({"frame_start":1,"fps":24.0,"width":0,"height":0}),
        ),
        "tracking.solve_object" => (
            vec!["id", "object", "name", "tracks"],
            vec!["op", "id", "object", "tracks"],
            "movie clip tracked object",
            "none",
            json!({}),
        ),
        "rig.generate_basic_human" => (
            vec!["target", "scale"],
            vec!["op", "target"],
            "armature object",
            "none",
            json!({"scale":1.0}),
        ),
        "shape_key.create" => (
            vec![
                "target",
                "id",
                "name",
                "positions",
                "value",
                "slider_min",
                "slider_max",
                "relative_key",
                "vertex_group",
                "frame",
            ],
            vec!["op", "target", "id", "name", "positions"],
            "mesh object",
            "shared|single_user for shared data",
            json!({"frame":0.0}),
        ),
        "shape_key.update" => (
            vec!["target", "id", "set"],
            vec!["op", "target", "id", "set"],
            "mesh object shape key",
            "shared|single_user for shared data",
            json!({}),
        ),
        "action.create" => (
            vec!["id", "name"],
            vec!["op", "id"],
            "action",
            "none",
            json!({"name":"id"}),
        ),
        "action.update" => (
            vec!["target", "set"],
            vec!["op", "target", "set"],
            "action",
            "none",
            json!({}),
        ),
        "action.delete" => (
            vec!["target"],
            vec!["op", "target"],
            "action",
            "none",
            json!({}),
        ),
        "keyframe.insert" => (
            vec!["target", "path", "index", "frame", "value", "interpolation"],
            vec!["op", "target", "path", "index", "frame", "value"],
            "object with assigned Action",
            "none",
            json!({"interpolation":"linear"}),
        ),
        "keyframe.delete" => (
            vec!["target", "path", "index", "frame"],
            vec!["op", "target", "path", "index", "frame"],
            "object with assigned Action",
            "none",
            json!({}),
        ),
        "fcurve.update" => (
            vec!["target", "path", "index", "set"],
            vec!["op", "target", "path", "index", "set"],
            "object with assigned Action F-Curve",
            "none",
            json!({}),
        ),
        "camera.create" => (
            vec![
                "id",
                "name",
                "tags",
                "collection",
                "parent",
                "parent_inverse",
                "transform",
                "visible",
                "render_visible",
                "selectable",
                "projection",
                "lens_mm",
                "sensor_width_mm",
                "ortho_scale",
                "clip_start",
                "clip_end",
                "shift",
                "panorama_type",
                "longitude_min",
                "longitude_max",
                "latitude_min",
                "latitude_max",
                "fisheye_fov",
                "fisheye_lens",
                "dof_enabled",
                "focus_distance",
                "f_stop",
                "aperture_blades",
                "stereo_mode",
                "interocular_distance",
            ],
            vec!["op", "id"],
            "camera object",
            "none",
            json!({"tags":[],"visible":true,"render_visible":true,"selectable":true,"projection":"perspective"}),
        ),
        "camera.update" => (
            vec!["target", "set"],
            vec!["op", "target", "set"],
            "camera object",
            "none",
            json!({}),
        ),
        "light.create" => (
            vec![
                "id",
                "name",
                "collection",
                "tags",
                "parent",
                "parent_inverse",
                "transform",
                "visible",
                "render_visible",
                "selectable",
                "light_type",
                "color",
                "energy",
                "radius",
                "soft_falloff",
                "use_multiple_importance_sampling",
                "spread",
                "spot_size",
                "spot_blend",
            ],
            vec!["op", "id"],
            "light object",
            "none",
            json!({"tags":[],"visible":true,"render_visible":true,"selectable":true,"light_type":"point","color":[1,1,1],"energy":1000,"radius":0.1}),
        ),
        "light.update" => (
            vec!["target", "set"],
            vec!["op", "target", "set"],
            "light object",
            "none",
            json!({}),
        ),
        "world.create" => (
            vec!["id", "color", "strength", "node_tree"],
            vec!["op", "id"],
            "world",
            "none",
            json!({"color":[1,1,1],"strength":1.0}),
        ),
        "world.update" => (
            vec!["target", "set"],
            vec!["op", "target", "set"],
            "world",
            "none",
            json!({}),
        ),
        "physics.world.update" | "simulation.settings.update" => (
            vec!["target", "set"],
            vec!["op", "target", "set"],
            "scene rigid body world",
            "none",
            json!({}),
        ),
        "physics.rigid_body.create" => (
            vec![
                "target",
                "type",
                "mass",
                "friction",
                "restitution",
                "shape",
                "linear_damping",
                "angular_damping",
                "initial_velocity",
            ],
            vec!["op", "target", "type"],
            "mesh object",
            "none",
            json!({"mass":1.0,"friction":0.5,"restitution":0.0,"shape":"box","linear_damping":0.04,"angular_damping":0.1,"initial_velocity":[0,0,0]}),
        ),
        "physics.rigid_body.update" => (
            vec!["target", "set"],
            vec!["op", "target", "set"],
            "mesh object rigid body",
            "none",
            json!({}),
        ),
        "physics.rigid_body.delete" => (
            vec!["target"],
            vec!["op", "target"],
            "mesh object rigid body",
            "none",
            json!({}),
        ),
        "physics.force_field.create" => (
            vec!["target", "type", "strength", "falloff"],
            vec!["op", "target", "type"],
            "Empty object",
            "none",
            json!({"strength":1.0,"falloff":0.0}),
        ),
        "physics.force_field.update" => (
            vec!["target", "set"],
            vec!["op", "target", "set"],
            "Empty object force field",
            "none",
            json!({}),
        ),
        "physics.force_field.delete" => (
            vec!["target"],
            vec!["op", "target"],
            "Empty object force field",
            "none",
            json!({}),
        ),
        "physics.cloth.create"
        | "physics.soft_body.create"
        | "physics.fluid.create"
        | "physics.dynamic_paint.create"
        | "physics.collision.create" => (
            vec!["target", "settings"],
            vec!["op", "target"],
            "mesh object physics system",
            "none",
            json!({}),
        ),
        "physics.particle_emitter.create" => (
            vec!["target", "settings", "seed"],
            vec!["op", "target"],
            "mesh object particle emitter",
            "none",
            json!({}),
        ),
        "physics.cloth.update"
        | "physics.soft_body.update"
        | "physics.particle_emitter.update"
        | "physics.fluid.update"
        | "physics.dynamic_paint.update"
        | "physics.collision.update" => (
            vec!["target", "set"],
            vec!["op", "target", "set"],
            "mesh object physics system",
            "none",
            json!({}),
        ),
        "physics.cloth.delete"
        | "physics.soft_body.delete"
        | "physics.particle_emitter.delete"
        | "physics.fluid.delete"
        | "physics.dynamic_paint.delete"
        | "physics.collision.delete" => (
            vec!["target"],
            vec!["op", "target"],
            "mesh object physics system",
            "none",
            json!({}),
        ),
        "avatar.clear" => (vec![], vec!["op"], "avatar settings", "none", json!({})),
        "avatar.update" => (
            vec!["set"],
            vec!["op", "set"],
            "avatar settings",
            "none",
            json!({"partial_components":true,"null_clears_optional_components":true}),
        ),
        _ => (
            vec!["target", "set"],
            vec!["op", "target", "set"],
            "operation-specific target",
            "none",
            json!({}),
        ),
    };
    OperationContract {
        fields,
        required,
        target_type,
        scope,
        reversible: true,
        defaults,
    }
}

fn field_schema(operation: &str, field: &str) -> Value {
    if operation == "avatar.update" && field == "set" {
        return avatar_update_set_schema();
    }
    if matches!(operation, "texture.create" | "texture.update") {
        return match field {
            "texture_type" if operation == "texture.create" => json!({"enum":[
                "CLOUDS","MARBLE","WOOD","MAGIC","BLEND","STUCCI","NOISE",
                "DISTORTED_NOISE","VORONOI","MUSGRAVE","NATIVE_VALUE_NOISE","IMAGE"
            ]}),
            "image" => json!({"oneOf":[id_schema(),{"type":"null"}]}),
            "settings" => json!({"type":"object"}),
            _ => common_field_schema(field),
        };
    }
    if matches!(
        operation,
        "library.link"
            | "library.append"
            | "library.override"
            | "library.register"
            | "library.relocate"
    ) {
        return match field {
            "target" if operation == "library.override" => target_schema(false, false),
            "kind" => json!({"enum":["blend"]}),
            "uri" | "resolved_path" | "hash" => json!({"type":"string"}),
            "resource" => json!({"oneOf":[{"$ref":"#/$defs/id"},{"type":"null"}]}),
            "items" if operation == "library.register" => json!({
                "type":"array",
                "minItems":1,
                "items":{
                    "type":"object",
                    "required":["registry","id","name"],
                    "properties":{
                        "registry":{"enum":["nodes","data_blocks","materials","collections","actions","node_groups","worlds","resources"]},
                        "id":{"$ref":"#/$defs/id"},
                        "name":{"type":"string"},
                        "source_id":{"$ref":"#/$defs/id"}
                    },
                    "additionalProperties":false
                }
            }),
            "items" => json!({
                "type":"array",
                "minItems":1,
                "items":{
                    "type":"object",
                    "required":["registry","id"],
                    "properties":{
                        "registry":{"enum":["nodes","data_blocks","materials","collections","actions","node_groups","worlds","resources"]},
                        "id":{"$ref":"#/$defs/id"}
                    },
                    "additionalProperties":false
                }
            }),
            "operations" => json!({
                "type":"array",
                "items":{
                    "type":"object",
                    "required":["op","path","value"],
                    "properties":{
                        "op":{"enum":["set","replace","insert_after","delete"]},
                        "path":{"type":"string"},
                        "value":{}
                    },
                    "additionalProperties":false
                }
            }),
            _ => common_field_schema(field),
        };
    }
    if operation == "library.reload" && field == "id" {
        return id_schema();
    }
    if operation == "tracking.solve_object" && field == "tracks" {
        return json!({
            "type":"array",
            "minItems":4,
            "items":{"type":"string","pattern":"^[a-z][a-z0-9_-]{0,63}$"},
            "uniqueItems":true
        });
    }
    if operation == "tracking.clip_create" {
        return match field {
            "source" | "source_hash" => json!({"type":["string","null"]}),
            "frame_start" => json!({"type":"integer"}),
            "fps" => json!({"type":"number","exclusiveMinimum":0}),
            "width" | "height" => json!({"type":"integer","minimum":0,"maximum":u32::MAX}),
            _ => common_field_schema(field),
        };
    }
    if field == "target" {
        if [
            "scene.update",
            "action.update",
            "action.delete",
            "world.update",
            "render.update",
            "physics.world.update",
            "simulation.settings.update",
            "physics.cloth.create",
            "physics.soft_body.create",
            "physics.particle_emitter.create",
            "physics.rigid_body.create",
            "physics.rigid_body.update",
            "physics.rigid_body.delete",
            "physics.force_field.create",
            "physics.force_field.update",
            "physics.force_field.delete",
            "physics.cloth.update",
            "physics.cloth.delete",
            "physics.soft_body.update",
            "physics.soft_body.delete",
            "physics.particle_emitter.update",
            "physics.particle_emitter.delete",
            "physics.fluid.create",
            "physics.fluid.update",
            "physics.fluid.delete",
            "physics.dynamic_paint.create",
            "physics.dynamic_paint.update",
            "physics.dynamic_paint.delete",
            "bone.create",
            "bone.update",
            "bone.delete",
            "bone_collection.create",
            "bone_collection.update",
            "bone_collection.delete",
            "bone_collection.assign",
            "bone_collection.unassign",
            "shape_key.create",
            "rig.generate_basic_human",
            "shape_key.update",
        ]
        .contains(&operation)
        {
            return json!({"type":"object","required":["id"],"properties":{"id":id_schema()},"additionalProperties":false});
        }
        let allow_many = ![
            "node.duplicate",
            "node.parent",
            "collection.parent",
            "keyframe.insert",
            "keyframe.delete",
            "fcurve.update",
            "camera.update",
            "light.update",
            "bone.create",
            "bone.update",
            "bone.delete",
            "bone_collection.create",
            "bone_collection.update",
            "bone_collection.delete",
            "bone_collection.assign",
            "bone_collection.unassign",
            "pose.set",
            "pose.reset",
            "constraint.create",
            "constraint.update",
            "constraint.delete",
            "constraint.reorder",
            "shape_key.create",
            "shape_key.update",
            "shape_key.delete",
            "vertex_group.create",
            "vertex_group.assign",
            "vertex_group.remove",
            "driver.create",
            "driver.update",
            "driver.delete",
        ]
        .contains(&operation);
        return target_schema(
            allow_many,
            operation.starts_with("mesh.") || operation.starts_with("uv."),
        );
    }
    if field == "elements" {
        return elements_schema();
    }
    if field == "settings" {
        return physics_settings_schema(operation);
    }
    if operation == "node.join" && field == "targets" {
        return json!({"type":"array","items":id_schema(),"minItems":1,"uniqueItems":true});
    }
    if operation == "mesh.separate" && field == "mode" {
        return json!({"enum":["selection","material","loose_parts"],"default":"selection"});
    }
    if operation == "mesh.attribute_create" && field == "domain" {
        return json!({"enum":["point","edge","face","corner"]});
    }
    if operation == "mesh.transform_elements" && field == "proportional" {
        return json!({
            "type":"object",
            "required":["radius"],
            "properties":{
                "radius":{"type":"number","exclusiveMinimum":0},
                "falloff":{"enum":["smooth","sphere","root","sharp","linear","constant"],"default":"smooth"},
                "connected_only":{"type":"boolean","default":false}
            },
            "additionalProperties":false
        });
    }
    if operation == "mesh.edge_slide" && field == "factor" {
        return json!({"type":"number","minimum":0,"maximum":1,"default":0.5});
    }
    if operation == "mesh.vertex_slide" && field == "factor" {
        return json!({"type":"number","minimum":-1,"maximum":1,"default":0});
    }
    if ["mesh.spin", "mesh.screw"].contains(&operation) {
        return match field {
            "axis" | "center" => vec_schema(3),
            "angle" => json!({"type":"number","description":"Rotation angle in radians."}),
            "steps" => json!({"type":"integer","minimum":1,"maximum":64,"default":9}),
            _ => common_field_schema(field),
        };
    }
    if operation == "mesh.merge" {
        return match field {
            "mode" => json!({"enum":["center","cursor"],"default":"center"}),
            "cursor" => vec_schema(3),
            _ => common_field_schema(field),
        };
    }
    if ["mesh.auto_smooth", "mesh.mark_sharp_by_angle"].contains(&operation) && field == "angle" {
        return json!({"type":"number","minimum":0,"maximum":std::f64::consts::PI,"default":std::f64::consts::FRAC_PI_4});
    }
    if operation == "mesh.set_custom_normals" && field == "normals" {
        return json!({"type":"object","propertyNames":{"pattern":"^f[0-9]+$"},"additionalProperties":{"type":"array","items":{"oneOf":[vec_schema(3),{"type":"null"}]}}});
    }
    if operation.starts_with("sculpt.") {
        return match field {
            "scope" => json!({"enum":["shared","single_user"]}),
            "brush" => {
                json!({"enum":["draw","clay_strips","inflate","grab","smooth","flatten","pinch","crease","layer","snake_hook","thumb","rotate","nudge","blob","scrape","fill","draw_sharp","elastic_deform","pose","boundary"]})
            }
            "falloff" => {
                json!({"enum":["smooth","smoother","sphere","root","sharp","linear","constant","pow4"],"description":"Falloff curve. Defaults to smoother for clay_strips, pow4 for draw_sharp, and smooth for other brushes."})
            }
            "deform_type" => {
                json!({"enum":["rotate_twist","scale_translate","squash_stretch","bend","expand","inflate","grab","twist","smooth"],"description":"Deform type of the pose or boundary brush. Pose: rotate_twist (default), scale_translate, or squash_stretch; pose twist is refused. Boundary: bend (default), expand, inflate, grab, twist, or smooth. Other brushes refuse it."})
            }
            "samples" => {
                json!({"type":"array","minItems":1,"items":{"type":"object","required":["position","pressure","radius","strength","time"],"properties":{"position":vec_schema(3),"pressure":{"type":"number","minimum":0,"maximum":1},"radius":{"type":"number","minimum":0},"strength":{"type":"number"},"time":{"type":"number"}},"additionalProperties":false}})
            }
            "symmetry" => json!({"type":"array","items":{"enum":["x","y","z"]},"uniqueItems":true}),
            "delta" | "view" => vec_schema(3),
            "dyntopo" => {
                json!({"type":"object","required":["edge_length"],"properties":{"edge_length":{"type":"number","exclusiveMinimum":0}},"additionalProperties":false})
            }
            "edge_length" => json!({"type":"number","exclusiveMinimum":0}),
            _ => common_field_schema(field),
        };
    }
    if field == "uv_layer" {
        return layer_name_schema();
    }
    if operation.starts_with("uv.") {
        return match field {
            "scope" => json!({"enum":["shared","single_user"]}),
            "method" => {
                json!({"enum":["box","smart","cube","angle_based","conformal"],"default":"box"})
            }
            "translation" | "scale" => vec_schema(2),
            "rotation" => json!({"type":"number","description":"Rotation angle in degrees."}),
            "pinned" => json!({"type":"boolean","default":true}),
            "margin" => json!({"type":"number","minimum":0,"maximum":0.5,"default":0.001}),
            "name" | "to" => layer_name_schema(),
            _ => common_field_schema(field),
        };
    }
    if ["camera.create", "camera.update"].contains(&operation) {
        return match field {
            "projection" => json!({"enum":["perspective","orthographic","panorama"]}),
            "panorama_type" => {
                json!({"enum":["equirectangular","fisheye_equidistant","fisheye_equisolid"]})
            }
            "longitude_min" | "longitude_max" => json!({"type":"number"}),
            "latitude_min" | "latitude_max" => {
                json!({"type":"number","minimum":-crate::model::BLENDER_RIGHT_ANGLE,"maximum":crate::model::BLENDER_RIGHT_ANGLE})
            }
            "fisheye_fov" => {
                json!({"type":"number","minimum":crate::model::FISHEYE_FOV_MIN,"maximum":crate::model::FISHEYE_FOV_MAX})
            }
            "fisheye_lens" => {
                json!({"type":"number","minimum":crate::model::FISHEYE_LENS_MIN,"maximum":crate::model::FISHEYE_LENS_MAX})
            }
            "stereo_mode" => json!({"enum":["none","side_by_side","anaglyph"]}),
            "dof_enabled" => json!({"type":"boolean"}),
            "focus_distance" | "f_stop" | "interocular_distance" => {
                json!({"type":"number","exclusiveMinimum":0})
            }
            "aperture_blades" => {
                json!({"oneOf":[{"const":0},{"type":"integer","minimum":3,"maximum":16}]})
            }
            _ => common_field_schema(field),
        };
    }
    if operation == "world.create" && field == "node_tree" {
        return json!({"oneOf":[id_schema(),{"type":"null"}]});
    }
    if ["material.create", "material.update"].contains(&operation) {
        return match field {
            "base_color_texture" | "roughness_texture" | "metallic_texture" | "normal_texture" => {
                nullable_texture_ref_schema()
            }
            "alpha_mode" => json!({"enum":["opaque","blend","clip"]}),
            "alpha_threshold" | "displacement_midlevel" => {
                json!({"type":"number","minimum":0,"maximum":1})
            }
            "displacement_method" => json!({"enum":["bump","displacement"]}),
            "volume_density" => json!({"type":"number","minimum":0}),
            "volume_color" => {
                json!({"type":"array","items":{"type":"number","minimum":0},"minItems":3,"maxItems":3})
            }
            "volume_anisotropy" => json!({"type":"number","minimum":-0.999,"maximum":0.999}),
            "displacement_scale" => json!({"type":"number"}),
            _ => common_field_schema(field),
        };
    }
    if operation == "constraint.update" && field == "set" {
        let constraint_types = params::type_names(ParameterFamily::Constraint);
        return json!({
            "type":"object",
            "properties":{
                "type":{"enum":constraint_types},
                "name":{"type":"string"},
                "target":{"oneOf":[id_schema(),{"type":"null"}]},
                "subtarget":{"oneOf":[id_schema(),{"type":"null"}]},
                "owner_bone":{"oneOf":[id_schema(),{"type":"null"}]},
                "influence":{"type":"number","minimum":0,"maximum":1},
                "enabled":{"type":"boolean"},
                "params":{"type":"object"},
                "set_inverse":{"type":"boolean"},
                "clear_inverse":{"type":"boolean"}
            },
            "additionalProperties":false,
            "minProperties":1
        });
    }
    if operation == "constraint.create" && field == "type" {
        return json!({"enum":params::type_names(ParameterFamily::Constraint)});
    }
    if operation == "constraint.create"
        && ["constraint_target", "subtarget", "owner_bone"].contains(&field)
    {
        return json!({"oneOf":[id_schema(),{"type":"null"}]});
    }
    match field {
        "parent"
            if [
                "node.create",
                "node.parent",
                "camera.create",
                "light.create",
                "collection.parent",
                "bone.create",
            ]
            .contains(&operation) =>
        {
            json!({"oneOf":[id_schema(),{"type":"null"}]})
        }
        "id" | "scene_id" | "data_id" | "parent" | "data" | "material" | "collection"
        | "object" | "child" | "root_collection" | "view_layer" | "modifier_id" | "action"
        | "bone_id" | "collection_id" => id_schema(),
        "bone_ids" | "tags" => json!({"type":"array","items":id_schema(),"uniqueItems":true}),
        "custom_shape" if operation.starts_with("bone.") => {
            json!({"oneOf":[{"$ref":"#/$defs/id"},{"type":"null"}]})
        }
        "scope" => json!({"enum":["shared","single_user"]}),
        "kind" if operation == "node.create" => json!({"enum":[
            "group","empty","mesh","camera","light","box","sphere","uv_sphere","cylinder",
            "plane","cone","torus","icosphere","circle","grid","curve","surface","text",
            "metaball","lattice","pointcloud","volume","armature","grease_pencil",
            "collection_instance"
        ]}),
        "track_axis" if operation.starts_with("node.") => {
            json!({"enum":["POS_X","POS_Y","POS_Z","NEG_X","NEG_Y","NEG_Z"]})
        }
        "up_axis" if operation.starts_with("node.") => json!({"enum":["X","Y","Z"]}),
        "kind" => {
            json!({"enum":["group","empty","mesh","camera","light","box","sphere","uv_sphere","cylinder","plane","cone","torus","icosphere","circle","grid"]})
        }
        "type" if operation == "modifier.create" => {
            json!({"enum":params::type_names(ParameterFamily::Modifier)})
        }
        "type" if operation == "mesh.attribute_create" => {
            json!({"enum":["float","int","float2","float3","color","bool","quaternion"]})
        }
        "type" if operation == "physics.force_field.create" => {
            json!({"enum":["wind","vortex","force"]})
        }
        "reparent" if operation == "node.delete" => json!({"enum":["to_parent","to_root"]}),
        "name" | "type" | "attribute" | "path" => json!({"type":"string"}),
        "mode" => json!({"enum":["independent","linked"],"default":"independent"}),
        "projection" => json!({"enum":["perspective","orthographic"],"default":"perspective"}),
        "light_type" => json!({"enum":["point","sun","spot","area"],"default":"point"}),
        "interpolation" => json!({"enum":["constant","linear","bezier"],"default":"linear"}),
        "method" => json!({"enum":["box","smart","cube"],"default":"box"}),
        "shape" => json!({"enum":["box","sphere","convex_hull","mesh"]}),
        "active" | "visible" | "render_visible" | "selectable" | "enabled" | "double_sided"
        | "show_viewport" | "show_render" | "recursive" | "cascade_data" | "unlink"
        | "keep_world" | "merge" | "individual" => {
            json!({"type":"boolean"})
        }
        "deform" | "inherit_rotation" | "use_connect" => json!({"type":"boolean"}),
        "absolute" if operation.starts_with("shape_key.") => json!({"type":"boolean"}),
        "parent_inverse" => {
            json!({"type":"array","items":{"type":"number"},"minItems":16,"maxItems":16})
        }
        "transform" => {
            json!({"type":"object","properties":{"translation":vec_schema(3),"rotation_deg":vec_schema(3),"rotation":vec_schema(4),"scale":vec_schema(3)},"additionalProperties":false})
        }
        "params" | "properties" => json!({"type":"object"}),
        "positions" if operation == "shape_key.create" => json!({"type":"object"}),
        "head" | "tail" if operation.starts_with("bone.") => vec_schema(3),
        "materials" => nullable_id_array_schema(),
        "base_color" => {
            json!({"type":"array","items":{"type":"number","minimum":0,"maximum":1},"minItems":4,"maxItems":4})
        }
        "color" | "emission_color" => {
            json!({"type":"array","items":{"type":"number","minimum":0},"minItems":3,"maxItems":3})
        }
        "ior" | "voxel_size" => json!({"type":"number","exclusiveMinimum":0}),
        "frame" | "evaluation_time" if operation.starts_with("shape_key.") => {
            json!({"type":"number"})
        }
        "envelope_distance" | "head_radius" | "tail_radius" if operation.starts_with("bone.") => {
            json!({"type":"number","minimum":0})
        }
        "envelope_weight" if operation.starts_with("bone.") => {
            json!({"type":"number","minimum":0,"maximum":1000})
        }
        "shift" => vec_schema(2),
        "strength" if operation == "physics.force_field.create" => json!({"type":"number"}),
        "mass" => json!({"type":"number","minimum":0,"default":1.0}),
        "friction" | "linear_damping" | "angular_damping" | "falloff" | "amount" | "distance"
        | "width" | "threshold" | "merge_threshold" => {
            json!({"type":"number","minimum":0})
        }
        "spot_blend" | "restitution" => json!({"type":"number","minimum":0,"maximum":1}),
        "scale" if operation == "rig.generate_basic_human" => {
            json!({"type":"number","exclusiveMinimum":0,"default":1.0})
        }
        "gravity" | "initial_velocity" | "translation" | "rotation" | "scale" | "normal"
        | "pivot" | "point" | "origin" | "offset" => vec_schema(3),
        "substeps" | "solver_iterations" => json!({"type":"integer","minimum":1,"maximum":128}),
        "seed" | "index" | "material_index" => {
            json!({"type":"integer","minimum":0,"maximum":u32::MAX})
        }
        "energy" | "radius" | "strength" => json!({"type":"number","minimum":0}),
        "spot_size" => {
            json!({"type":"number","exclusiveMinimum":0,"maximum":std::f64::consts::TAU})
        }
        "value" => json!({}),
        "frame_start" | "frame_end" => {
            json!({"type":"integer","minimum":i32::MIN,"maximum":i32::MAX})
        }
        "segments" if operation == "mesh.bevel" => json!({"const":1,"default":1}),
        "cuts" if operation == "mesh.loop_cut" || operation == "mesh.subdivide" => {
            json!({"type":"integer","minimum":1,"maximum":8})
        }
        "segments" | "cuts" => json!({"type":"integer","minimum":1}),
        "axis" => json!({"enum":["x","y","z"]}),
        "direction" => json!({"enum":["positive_to_negative","negative_to_positive"]}),
        "plane" => {
            json!({"type":"object","properties":{"point":vec_schema(3),"normal":vec_schema(3)},"required":["point","normal"],"additionalProperties":false})
        }
        "clear_side" => json!({"enum":["positive","negative","none"]}),
        "set" => set_schema(operation),
        "unit" => {
            json!({"type":"object","properties":{"system":{"type":"string"},"scale_length":{"type":"number","exclusiveMinimum":0}},"additionalProperties":false})
        }
        _ => common_field_schema(field),
    }
}

fn common_field_schema(field: &str) -> Value {
    match field {
        "root_collection" | "scene_id" | "view_layer" | "id" | "modifier_id" | "collection"
        | "object" | "child" | "data_id" => id_schema(),
        "parent" => json!({"oneOf":[id_schema(),{"type":"null"}]}),
        "tags" => json!({"type":"array","items":id_schema(),"uniqueItems":true}),
        "name" => json!({"type":"string"}),
        "visible" | "render_visible" | "selectable" => json!({"type":"boolean"}),
        "parent_inverse" => {
            json!({"type":"array","items":{"type":"number"},"minItems":16,"maxItems":16})
        }
        "transform" => {
            json!({"type":"object","properties":{"translation":vec_schema(3),"rotation_deg":vec_schema(3),"rotation":vec_schema(4),"scale":vec_schema(3)},"additionalProperties":false})
        }
        "pivot" | "origin" | "point" | "offset" | "normal" | "translation" | "rotation"
        | "scale" => vec_schema(3),
        _ => json!({}),
    }
}

fn target_schema(allow_many: bool, allow_elements: bool) -> Value {
    let mut props = Map::new();
    props.insert("id".to_owned(), id_schema());
    props.insert("tag".to_owned(), id_schema());
    if allow_many {
        props.insert("many".to_owned(), json!({"type":"boolean"}));
    }
    if allow_elements {
        props.insert("elements".to_owned(), elements_schema());
    }
    let id = json!({"type":"object","required":["id"],"properties":props.clone(),"additionalProperties":false,"not":{"anyOf":[{"required":["tag"]},{"required":["many"]}]}});
    let tag = json!({"type":"object","required":["tag"],"properties":props,"additionalProperties":false,"not":{"required":["id"]}});
    json!({"oneOf":[id,tag]})
}

fn elements_schema() -> Value {
    json!({
        "type":"object",
        "required":["domain"],
        "properties":{
            "domain":{"enum":["vertex","point","edge","face"]},
            "ids":{"type":"array","items":{"type":["integer","string"]},"minItems":1,"uniqueItems":true},
            "selector":{
                "oneOf":[
                    {"type":"object","required":["type","min","max"],"properties":{"type":{"const":"position_box"},"min":vec_schema(3),"max":vec_schema(3)},"additionalProperties":false},
                    {"type":"object","required":["type","center","radius"],"properties":{"type":{"const":"position_sphere"},"center":vec_schema(3),"radius":{"type":"number","minimum":0}},"additionalProperties":false},
                    {"type":"object","required":["type","axis","angle"],"properties":{"type":{"const":"normal_cone"},"axis":vec_schema(3),"angle":{"type":"number","minimum":0,"maximum":std::f64::consts::PI}},"additionalProperties":false},
                    {"type":"object","required":["type","name","value"],"properties":{"type":{"const":"attribute"},"name":{"type":"string","minLength":1},"operator":{"enum":["equals","less_than","less_than_or_equal","greater_than","greater_than_or_equal"],"default":"equals"},"value":{}},"additionalProperties":false},
                    {"type":"object","required":["type","ids"],"properties":{"type":{"enum":["connected","linked"]},"ids":{"type":"array","items":{"type":"string"},"minItems":1,"uniqueItems":true}},"additionalProperties":false},
                    {"type":"object","required":["type","index"],"properties":{"type":{"const":"material_index"},"index":{"type":"integer","minimum":0,"maximum":u32::MAX}},"additionalProperties":false}
                ]
            }
        },
        "oneOf":[{"required":["ids"]},{"required":["selector"]}],
        "additionalProperties":false
    })
}

fn set_schema(operation: &str) -> Value {
    let mut properties = match operation {
        "scene.update" => json!({
            "name":{"type":"string"},"frame_current":{"type":"number"},
            "frame_start":{"type":"integer"},"frame_end":{"type":"integer"},
            "fps":{"type":"integer","minimum":1},"fps_base":{"type":"number","exclusiveMinimum":0},
            "camera":{"oneOf":[{"$ref":"#/$defs/id"},{"type":"null"}]},
            "world":{"oneOf":[{"$ref":"#/$defs/id"},{"type":"null"}]},
            "clip":{"oneOf":[{"$ref":"#/$defs/id"},{"type":"null"}]},
            "root_collection":{"$ref":"#/$defs/id"},
            "unit":{"type":"object","properties":{"system":{"type":"string"},"scale_length":{"type":"number","exclusiveMinimum":0}},"additionalProperties":false}
        }),
        "collection.update" => json!({"name":{"type":"string"},"exclude":{"type":"boolean"}}),
        "node.update" => json!({
            "name":{"type":"string"},"tags":{"type":"array","items":id_schema()},
            "transform":{"type":"object","properties":{"translation":vec_schema(3),"rotation_deg":vec_schema(3),"rotation":vec_schema(4),"scale":vec_schema(3)},"additionalProperties":false},
            "params":{"type":"object"},"materials":{"type":"array","items":{"oneOf":[id_schema(),{"type":"null"}]}},
            "visible":{"type":"boolean"},"render_visible":{"type":"boolean"},"selectable":{"type":"boolean"},
            "track_axis":{"enum":["POS_X","POS_Y","POS_Z","NEG_X","NEG_Y","NEG_Z"]},"up_axis":{"enum":["X","Y","Z"]},
            "action":{"oneOf":[{"$ref":"#/$defs/id"},{"type":"null"}]}
        }),
        "shape_key.update" => json!({
            "name":{"type":"string"},
            "positions":{"type":"object"},
            "value":{"type":"number"},
            "slider_min":{"type":"number"},
            "slider_max":{"type":"number"},
            "relative_key":{"oneOf":[{"$ref":"#/$defs/id"},{"type":"null"}]},
            "vertex_group":{"oneOf":[{"$ref":"#/$defs/id"},{"type":"null"}]},
            "frame":{"type":"number"},
            "evaluation_time":{"type":"number"},
            "absolute":{"type":"boolean"}
        }),
        "bone.update" => json!({
            "name":{"type":"string"},
            "head":vec_schema(3),
            "tail":vec_schema(3),
            "parent":{"oneOf":[{"$ref":"#/$defs/id"},{"type":"null"}]},
            "roll":{"type":"number"},
            "deform":{"type":"boolean"},
            "inherit_rotation":{"type":"boolean"},
            "use_connect":{"type":"boolean"},
            "custom_shape":{"oneOf":[{"$ref":"#/$defs/id"},{"type":"null"}]},
            "envelope_distance":{"type":"number","minimum":0},
            "envelope_weight":{"type":"number","minimum":0,"maximum":1000},
            "head_radius":{"type":"number","minimum":0},
            "tail_radius":{"type":"number","minimum":0}
        }),
        "bone_collection.update" => json!({
            "name":{"type":"string"},
            "visible":{"type":"boolean"}
        }),
        "material.update" => material_set_properties(),
        "modifier.update" => {
            json!({"name":{"type":"string"},"enabled":{"type":"boolean"},"show_viewport":{"type":"boolean"},"show_render":{"type":"boolean"},"type":{"enum":params::type_names(ParameterFamily::Modifier)},"params":{"type":"object"}})
        }
        "action.update" => {
            json!({"name":{"type":"string"},"fcurves":{"type":"array","items":{"$ref":"#/$defs/fcurve"}}})
        }
        "fcurve.update" => json!({"extrapolation":{"enum":["constant","linear"]}}),
        "camera.update" => camera_set_properties(),
        "light.update" => light_set_properties(),
        "world.update" => world_set_properties(),
        "physics.world.update" | "simulation.settings.update" => rigid_body_world_properties(),
        "physics.rigid_body.update" => rigid_body_properties(),
        "physics.force_field.update" => force_field_properties(),
        "physics.cloth.update"
        | "physics.soft_body.update"
        | "physics.particle_emitter.update"
        | "physics.fluid.update"
        | "physics.dynamic_paint.update" => physics_settings_properties(operation),
        "render.update" => json!({
            "resolution_x":{"type":"integer","minimum":1,"maximum":16384},
            "resolution_y":{"type":"integer","minimum":1,"maximum":16384},
            "resolution_percentage":{"type":"integer","minimum":1,"maximum":100},
            "samples":{"type":"integer","minimum":1,"maximum":1_000_000},
            "seed":{"type":"integer","minimum":0,"maximum":u32::MAX},
            "max_bounces":{"type":"integer","minimum":0,"maximum":1024},
            "transparent_max_bounces":{"type":"integer","minimum":0,"maximum":1024},
            "film_transparent":{"type":"boolean"},
            "engine":{"enum":["path","realtime"]},
            "use_sequencer":{"type":"boolean"},
            "audio_codec":{"enum":["wav","flac"]},
            "motion_blur":{"type":"boolean"},
            "shutter":{"type":"number","exclusiveMinimum":0,"maximum":2},
            "motion_blur_samples":{"type":"integer","minimum":1,"maximum":64}
        }),
        _ => json!({}),
    };
    if operation == "node.update" {
        properties["materials"] = nullable_id_array_schema();
    }

    json!({"type":"object","properties":properties,"minProperties":1,"additionalProperties":false})
}

fn physics_settings_schema(operation: &str) -> Value {
    let properties = physics_settings_properties(operation);
    json!({"type":"object","properties":properties,"additionalProperties":false})
}

fn physics_settings_properties(operation: &str) -> Value {
    let system = operation
        .strip_prefix("physics.")
        .and_then(|name| name.split_once('.'))
        .map(|(system, _)| system)
        .unwrap_or_default();
    match system {
        "cloth" => json!({
            "stiffness":{"type":"number","minimum":0,"maximum":1},
            "structural_stiffness":{"type":"number","minimum":0,"maximum":1},
            "shear_stiffness":{"type":"number","minimum":0,"maximum":1},
            "bend_stiffness":{"type":"number","minimum":0,"maximum":1},
            "air_drag":{"type":"number","minimum":0},"drag":{"type":"number","minimum":0},
            "damping":{"type":"number","minimum":0},
            "substeps":{"type":"integer","minimum":1,"maximum":32},
            "iterations":{"type":"integer","minimum":1,"maximum":64},
            "quality":{"type":"integer","minimum":1,"maximum":32},
            "pin_group":{"type":"string","minLength":1},
            "tension_stiffness":{"type":"number","minimum":0,"maximum":1},
            "compression_stiffness":{"type":"number","minimum":0,"maximum":1},
            "bending_stiffness":{"type":"number","minimum":0,"maximum":1},
            "mass":{"type":"number","exclusiveMinimum":0},
            "air_damping":{"type":"number","minimum":0},
            "vertex_group_mass":{"type":"string","minLength":1},
            "use_collision":{"type":"boolean"},
            "collision_distance":{"type":"number","minimum":0},
            "use_self_collision":{"type":"boolean"},
            "self_collision_distance":{"type":"number","minimum":0},
            "distance_min":{"type":"number","minimum":0},
        }),
        "soft_body" => json!({
            "stiffness":{"type":"number","minimum":0,"maximum":1},
            "edge_stiffness":{"type":"number","minimum":0,"maximum":1},
            "volume_stiffness":{"type":"number","minimum":0,"maximum":1},
            "goal_group":{"type":"string","minLength":1},
            "vertex_group_goal":{"type":"string","minLength":1},
            "goal_strength":{"type":"number","minimum":0,"maximum":1},
            "goal_stiffness":{"type":"number","minimum":0,"maximum":1},
            "substeps":{"type":"integer","minimum":1,"maximum":32},
            "iterations":{"type":"integer","minimum":1,"maximum":64},
            "quality":{"type":"integer","minimum":1,"maximum":32},
            "air_drag":{"type":"number","minimum":0},"drag":{"type":"number","minimum":0},
            "damping":{"type":"number","minimum":0},
            "restitution":{"type":"number","minimum":0,"maximum":1},
            "goal_spring":{"type":"number","minimum":0,"maximum":1},
            "goal_friction":{"type":"number","minimum":0,"maximum":1},
            "goal_default":{"type":"number","minimum":0,"maximum":1},
            "goal_min":{"type":"number"},
            "goal_max":{"type":"number"},
            "mass":{"type":"number","exclusiveMinimum":0},
            "friction":{"type":"number","minimum":0},
            "speed":{"type":"number","minimum":0},
            "plastic":{"type":"number","minimum":0,"maximum":1},
            "bend":{"type":"number","minimum":0,"maximum":1},
        }),
        "particle_emitter" => json!({
            "seed":{"type":"integer","minimum":0,"maximum":u32::MAX},
            "type":{"enum":["EMITTER","HAIR"]},
            "child_seed":{"type":"integer","minimum":0,"maximum":u32::MAX},
            "lifetime":{"type":"number","exclusiveMinimum":0},
            "speed":{"type":"number","minimum":0},
            "random_velocity":{"oneOf":[{"type":"number","minimum":0},vec_schema(3)]},
            "count":{"type":"integer","minimum":0,"maximum":1_000_000},
            "lifetime_random":{"type":"number","minimum":0,"maximum":1},
            "normal_factor":{"type":"number","minimum":0},
            "factor_random":{"type":"number","minimum":0},
            "emit_from":{"enum":["VERT","FACE","VOLUME"]},
            "physics_type":{"enum":["NEWTON","NO"]},
            "mass":{"type":"number","exclusiveMinimum":0},
            "render_type":{"enum":["HALO","PATH","OBJECT","COLLECTION"]},
            "instance_collection":id_schema(),
            "particle_size":{"type":"number","exclusiveMinimum":0},
            "size_random":{"type":"number","minimum":0,"maximum":1},
            "child_percent":{"type":"integer","minimum":0,"maximum":100000},
            "rendered_child_count":{"type":"integer","minimum":0,"maximum":100000},
            "child_radius":{"type":"number","minimum":0,"maximum":100000},
            "child_type":{"enum":["NONE","SIMPLE","INTERPOLATED"]},
            "child_size":{"type":"number","exclusiveMinimum":0,"maximum":100000},
            "child_size_random":{"type":"number","minimum":0,"maximum":1},
            "child_roundness":{"type":"number","minimum":0,"maximum":1},
            "virtual_parents":{"type":"number","minimum":0,"maximum":1},
            "use_emit_random":{"type":"boolean"},
            "frame_start":{"type":"number"},
            "frame_end":{"type":"number"}
        }),
        "fluid" => json!({
            "type":{"enum":["liquid","smoke","fire"]},
            "resolution":{"type":"integer","minimum":1,"maximum":10},
            "particle_radius":{"type":"number","exclusiveMinimum":0},
            "smoothing_length":{"type":"number","exclusiveMinimum":0},
            "pressure_stiffness":{"type":"number","minimum":0},
            "viscosity":{"type":"number","minimum":0},
            "inflow_rate":{"type":"number","minimum":0},
            "seed":{"type":"integer","minimum":0,"maximum":u32::MAX}
        }),
        "dynamic_paint" => json!({
            "role":{"enum":["canvas","brush"]},
            "radius":{"type":"number","exclusiveMinimum":0},
            "color":{"type":"array","items":{"type":"number"},"minItems":4,"maxItems":4},
            "strength":{"type":"number","minimum":0,"maximum":1},
            "brushes":{"type":"array","items":id_schema()},
            "surface_format":{"enum":["color","weight"]}
        }),
        "collision" => json!({
            "thickness":{"type":"number","minimum":0},
            "thickness_outer":{"type":"number","minimum":0},
            "thickness_inner":{"type":"number","minimum":0},
            "damping":{"type":"number","minimum":0},
            "damping_factor":{"type":"number","minimum":0,"maximum":1},
            "damping_random":{"type":"number","minimum":0,"maximum":1},
            "permeability":{"type":"number","minimum":0,"maximum":1},
            "stickiness":{"type":"number","minimum":0,"maximum":1},
            "friction_factor":{"type":"number","minimum":0,"maximum":1},
            "friction_random":{"type":"number","minimum":0,"maximum":1},
            "use_culling":{"type":"boolean"},
            "use_normal":{"type":"boolean"},
            "use_particle_kill":{"type":"boolean"}
        }),
        _ => json!({}),
    }
}
fn material_set_properties() -> Value {
    json!({
        "name":{"type":"string"},
        "alpha_mode":{"enum":["opaque","blend","clip"]},
        "alpha_threshold":{"type":"number","minimum":0,"maximum":1},
        "base_color":{"type":"array","items":{"type":"number","minimum":0,"maximum":1},"minItems":4,"maxItems":4},
        "metallic":{"type":"number","minimum":0,"maximum":1},
        "roughness":{"type":"number","minimum":0,"maximum":1},
        "double_sided":{"type":"boolean"},
        "use_transparent_shadow":{"type":"boolean"},
        "emission_color":{"type":"array","items":{"type":"number","minimum":0},"minItems":3,"maxItems":3},
        "emission_strength":{"type":"number","minimum":0},
        "transmission":{"type":"number","minimum":0,"maximum":1},
        "ior":{"type":"number","exclusiveMinimum":0},
        "base_color_texture":nullable_texture_ref_schema(),
        "roughness_texture":nullable_texture_ref_schema(),
        "metallic_texture":nullable_texture_ref_schema(),
        "normal_texture":nullable_texture_ref_schema(),
        "displacement_method":{"enum":["bump","displacement"]},
        "volume_density":{"type":"number","minimum":0},
        "volume_color":{"type":"array","items":{"type":"number","minimum":0},"minItems":3,"maxItems":3},
        "volume_anisotropy":{"type":"number","minimum":-0.999,"maximum":0.999},
        "displacement_scale":{"type":"number"},
        "displacement_midlevel":{"type":"number","minimum":0,"maximum":1}
    })
}

fn camera_set_properties() -> Value {
    json!({
        "projection":{"enum":["perspective","orthographic","panorama"]},
        "lens_mm":{"type":"number","exclusiveMinimum":0},
        "sensor_width_mm":{"type":"number","exclusiveMinimum":0},
        "ortho_scale":{"type":"number","exclusiveMinimum":0},
        "clip_start":{"type":"number","exclusiveMinimum":0},
        "clip_end":{"type":"number","exclusiveMinimum":0},
        "shift":vec_schema(2),
        "panorama_type":{"enum":["equirectangular","fisheye_equidistant","fisheye_equisolid"]},
        "longitude_min":{"type":"number"},
        "longitude_max":{"type":"number"},
        "latitude_min":{"type":"number","minimum":-crate::model::BLENDER_RIGHT_ANGLE,"maximum":crate::model::BLENDER_RIGHT_ANGLE},
        "latitude_max":{"type":"number","minimum":-crate::model::BLENDER_RIGHT_ANGLE,"maximum":crate::model::BLENDER_RIGHT_ANGLE},
        "fisheye_fov":{"type":"number","minimum":crate::model::FISHEYE_FOV_MIN,"maximum":crate::model::FISHEYE_FOV_MAX},
        "fisheye_lens":{"type":"number","minimum":crate::model::FISHEYE_LENS_MIN,"maximum":crate::model::FISHEYE_LENS_MAX},
        "dof_enabled":{"type":"boolean"},
        "focus_distance":{"type":"number","exclusiveMinimum":0},
        "f_stop":{"type":"number","exclusiveMinimum":0},
        "aperture_blades":{"oneOf":[{"const":0},{"type":"integer","minimum":3,"maximum":16}]},
        "stereo_mode":{"enum":["none","side_by_side","anaglyph"]},
        "interocular_distance":{"type":"number","exclusiveMinimum":0}
    })
}

/// The schema of camera data as a scene document stores it: every field `CameraData` serialises, with the ranges the camera
/// operations accept.
fn camera_data_schema() -> Value {
    let mut properties = camera_set_properties();
    properties["sensor_height_mm"] = json!({"type":"number","exclusiveMinimum":0});
    properties["sensor_fit"] = json!({"enum":["AUTO","HORIZONTAL","VERTICAL"]});
    json!({"type":"object","properties":properties,"additionalProperties":false})
}

fn light_set_properties() -> Value {
    json!({"light_type":{"enum":["point","sun","spot","area"]},"color":{"type":"array","items":{"type":"number","minimum":0},"minItems":3,"maxItems":3},"energy":{"type":"number","minimum":0},"radius":{"type":"number","minimum":0},"soft_falloff":{"type":"boolean"},"use_multiple_importance_sampling":{"type":"boolean"},"spread":{"type":"number","minimum":0,"maximum":std::f64::consts::PI},"spot_size":{"type":"number","exclusiveMinimum":0,"maximum":std::f64::consts::TAU},"spot_blend":{"type":"number","minimum":0,"maximum":1}})
}

fn world_set_properties() -> Value {
    json!({
        "color":{"type":"array","items":{"type":"number","minimum":0},"minItems":3,"maxItems":3},
        "strength":{"type":"number","minimum":0},
        "node_tree":{"oneOf":[id_schema(),{"type":"null"}]}
    })
}
fn rigid_body_world_properties() -> Value {
    json!({"enabled":{"type":"boolean"},"gravity":vec_schema(3),"substeps":{"type":"integer","minimum":1,"maximum":128},"solver_iterations":{"type":"integer","minimum":1,"maximum":128},"frame_start":{"type":"integer","minimum":i32::MIN,"maximum":i32::MAX},"frame_end":{"type":"integer","minimum":i32::MIN,"maximum":i32::MAX},"seed":{"type":"integer","minimum":0,"maximum":u32::MAX}})
}

fn rigid_body_properties() -> Value {
    json!({"type":{"enum":["active","passive"]},"mass":{"type":"number","minimum":0},"friction":{"type":"number","minimum":0},"restitution":{"type":"number","minimum":0,"maximum":1},"shape":{"enum":["box","sphere","convex_hull","mesh"]},"linear_damping":{"type":"number","minimum":0},"angular_damping":{"type":"number","minimum":0},"initial_velocity":vec_schema(3)})
}

fn force_field_properties() -> Value {
    json!({"type":{"enum":["wind","vortex","force"]},"strength":{"type":"number"},"falloff":{"type":"number","minimum":0}})
}

fn def_key(name: &str) -> String {
    format!("op_{}", name.replace('.', "_"))
}

fn preview_schema() -> Value {
    json!({"$schema":DRAFT,"title":"Potter preview manifest","type":"object","required":["schema_version","scene_id","revision","scene_hash","evaluation_hash","files"],"properties":{"schema_version":{"const":1},"scene_id":{"type":"string"},"revision":{"type":"integer","minimum":0},"scene_hash":{"type":"string"},"evaluation_hash":{"type":"string"},"files":{"type":"array","items":{"type":"object"}},"camera":{"type":["string","null"]},"view_layer":{"type":["string","null"]},"frame":{"type":"number"}},"additionalProperties":true})
}

fn response_schema() -> Value {
    json!({"$schema":DRAFT,"title":"Potter command response","type":"object","required":["schema_version","command","ok","scene","result","warnings","error"],"properties":{"schema_version":{"const":1},"command":{"type":["string","null"]},"ok":{"type":"boolean"},"scene":{"type":["object","null"]},"result":{},"warnings":{"type":"array","items":{"type":"object","required":["code","message"],"properties":{"code":{"type":"string"},"message":{"type":"string"},"data_id":{"type":["string","null"]},"details":{}},"additionalProperties":true}},"error":{"type":["object","null"]}},"additionalProperties":false})
}

#[cfg(test)]
mod tests {
    use super::{operations_schema, schema};

    #[test]
    fn all_dispatch_operations_have_a_schema() {
        for name in crate::ops::OP_NAMES {
            let operation = schema("operations", Some(name)).unwrap_or(Value::Null);
            assert_eq!(operation["properties"]["op"]["const"], *name);
        }
        let document = operations_schema();
        assert_eq!(
            document["$defs"].as_object().map(serde_json::Map::len),
            Some(crate::ops::OP_NAMES.len() + 3)
        );
    }

    #[test]
    fn pose_constraint_owner_bone_is_in_scene_and_operation_schemas() -> crate::error::Result<()> {
        let scene = schema("scene", None)?;
        let owner_bone = &scene["$defs"]["constraint"]["properties"]["owner_bone"];
        assert_eq!(owner_bone["oneOf"][0]["$ref"], "#/$defs/id");
        assert_eq!(owner_bone["oneOf"][1]["type"], "null");
        let create = schema("operations", Some("constraint.create"))?;
        assert_eq!(
            create["properties"]["owner_bone"]["oneOf"][0]["pattern"],
            "^[a-z][a-z0-9_-]{0,63}$"
        );
        let update = schema("operations", Some("constraint.update"))?;
        assert_eq!(
            update["properties"]["set"]["properties"]["owner_bone"]["oneOf"][0]["pattern"],
            "^[a-z][a-z0-9_-]{0,63}$"
        );
        Ok(())
    }
    #[test]
    fn modifier_and_constraint_schemas_use_rna_parameter_tables() -> crate::error::Result<()> {
        let modifier_create = schema("operations", Some("modifier.create"))?;
        let array_params = modifier_create["allOf"]
            .as_array()
            .and_then(|conditions| {
                conditions
                    .iter()
                    .find(|condition| condition["if"]["properties"]["type"]["const"] == "array")
            })
            .map(|condition| &condition["then"]["properties"]["params"]);
        assert!(array_params.is_some_and(|params| {
            params["additionalProperties"] == false
                && params["properties"]["relative_offset_displace"]["type"] == "array"
                && params["properties"]["relative_offset_displace"]["default"]
                    == serde_json::json!([1.0, 0.0, 0.0])
                && params["properties"].get("relative_offset").is_none()
        }));

        let modifier_update = schema("operations", Some("modifier.update"))?;
        let updated_array_params = modifier_update["allOf"]
            .as_array()
            .and_then(|conditions| {
                conditions.iter().find(|condition| {
                    condition["if"]["properties"]["set"]["properties"]["type"]["const"] == "array"
                })
            })
            .map(|condition| &condition["then"]["properties"]["set"]["properties"]["params"]);
        assert!(updated_array_params.is_some_and(|params| {
            params["additionalProperties"] == false
                && params["properties"]
                    .get("constant_offset_displace")
                    .is_some()
        }));

        let constraint_create = schema("operations", Some("constraint.create"))?;
        let track_to_params = constraint_create["allOf"]
            .as_array()
            .and_then(|conditions| {
                conditions
                    .iter()
                    .find(|condition| condition["if"]["properties"]["type"]["const"] == "track_to")
            })
            .map(|condition| &condition["then"]["properties"]["params"]);
        assert!(track_to_params.is_some_and(|params| {
            params["additionalProperties"] == false
                && params["properties"]["track_axis"]["enum"]
                    .as_array()
                    .is_some_and(|values| values.iter().any(|value| value == "TRACK_Z"))
        }));
        Ok(())
    }

    #[test]
    fn attribute_create_type_accepts_color_and_rejects_byte_color() -> crate::error::Result<()> {
        let create = schema("operations", Some("mesh.attribute_create"))?;
        let types = &create["properties"]["type"]["enum"];
        assert!(
            types
                .as_array()
                .is_some_and(|types| types.iter().any(|value| value == "color"))
        );
        assert!(
            types
                .as_array()
                .is_some_and(|types| types.iter().all(|value| value != "byte_color"))
        );
        Ok(())
    }

    #[test]
    fn texture_image_operations_schema_reference_shared_images() -> crate::error::Result<()> {
        let create = schema("operations", Some("texture.create"))?;
        assert_eq!(
            create["properties"]["image"]["oneOf"][0]["pattern"],
            "^[a-z][a-z0-9_-]{0,63}$"
        );
        assert_eq!(create["properties"]["image"]["oneOf"][1]["type"], "null");
        assert!(
            create["properties"]["texture_type"]["enum"]
                .as_array()
                .is_some_and(|types| types.iter().any(|texture_type| texture_type == "IMAGE"))
        );
        assert!(
            create["allOf"][0]["then"]["required"]
                .as_array()
                .is_some_and(|required| required.iter().any(|field| field == "image"))
        );
        assert_eq!(
            create["allOf"][0]["then"]["properties"]["settings"]["additionalProperties"],
            false
        );
        let update = schema("operations", Some("texture.update"))?;
        assert_eq!(
            update["properties"]["image"]["oneOf"][0]["pattern"],
            "^[a-z][a-z0-9_-]{0,63}$"
        );
        assert_eq!(update["properties"]["image"]["oneOf"][1]["type"], "null");
        Ok(())
    }
    use serde_json::Value;
}
