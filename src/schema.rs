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
        "schema_version":{"const":1},
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
        "resources":{"type":"object"},
        "libraries":registry_schema("library"),
        "compatibility":{"type":"object"},
        "history":{"$ref":"#/$defs/history"}
    });
    root["additionalProperties"] = json!(false);
    root["$defs"] = json!({
        "id":id_schema(),
        "vec2":vec_schema(2),
        "vec3":vec_schema(3),
        "profile":{"type":"object","required":["blender"],"properties":{"blender":{"type":"string"}},"additionalProperties":false},
        "unit":{"type":"object","required":["system","scale_length"],"properties":{"system":{"type":"string"},"scale_length":{"type":"number","exclusiveMinimum":0}},"additionalProperties":false},
        "view_layer":{"type":"object","required":["name","excluded_collections"],"properties":{"name":{"type":"string"},"excluded_collections":{"type":"array","items":{"$ref":"#/$defs/id"}}},"additionalProperties":false},
        "scene":{"type":"object","required":["name","root_collection","view_layers","frame_current","frame_start","frame_end","fps","fps_base","camera","world","unit","markers","render","rigid_body_world"],"properties":{"name":{"type":"string"},"root_collection":{"$ref":"#/$defs/id"},"view_layers":registry_schema("view_layer"),"frame_current":{"type":"number"},"frame_start":{"type":"integer"},"frame_end":{"type":"integer"},"fps":{"type":"integer","minimum":1},"fps_base":{"type":"number","exclusiveMinimum":0},"camera":{"oneOf":[{"$ref":"#/$defs/id"},{"type":"null"}]},"world":{"oneOf":[{"$ref":"#/$defs/id"},{"type":"null"}]},"unit":{"$ref":"#/$defs/unit"},"render":{"$ref":"#/$defs/render_settings"},"markers":{"type":"array","items":{"$ref":"#/$defs/marker"}},"rigid_body_world":{"oneOf":[{"$ref":"#/$defs/rigid_body_world"},{"type":"null"}]}},"additionalProperties":false},
        "render_settings":{"type":"object","required":["resolution_x","resolution_y","resolution_percentage","samples","seed","max_bounces","film_transparent","engine","use_sequencer","audio_codec","passes"],"properties":{"resolution_x":{"type":"integer","minimum":1,"maximum":16384},"resolution_y":{"type":"integer","minimum":1,"maximum":16384},"resolution_percentage":{"type":"integer","minimum":1,"maximum":100},"samples":{"type":"integer","minimum":1,"maximum":1_000_000},"seed":{"type":"integer","minimum":0,"maximum":u32::MAX},"max_bounces":{"type":"integer","minimum":0,"maximum":1024},"film_transparent":{"type":"boolean"},"engine":{"enum":["path","realtime"]},"use_sequencer":{"type":"boolean"},"audio_codec":{"enum":["wav","flac"]},"passes":{"type":"array","items":{"type":"string"}}},"additionalProperties":false},
        "marker":{"type":"object","required":["id","name","frame"],"properties":{"id":{"$ref":"#/$defs/id"},"name":{"type":"string"},"frame":{"type":"number"}},"additionalProperties":false},
        "rigid_body_world":{"type":"object","properties":{"enabled":{"type":"boolean"},"gravity":{"$ref":"#/$defs/vec3"},"substeps":{"type":"integer","minimum":1,"maximum":128},"solver_iterations":{"type":"integer","minimum":1,"maximum":128},"frame_start":{"type":"integer","minimum":i32::MIN,"maximum":i32::MAX},"frame_end":{"type":"integer","minimum":i32::MIN,"maximum":i32::MAX},"seed":{"type":"integer","minimum":0,"maximum":u32::MAX}},"additionalProperties":false},
        "rigid_body":{"type":"object","properties":{"type":{"enum":["active","passive"]},"mass":{"type":"number","minimum":0},"friction":{"type":"number","minimum":0},"restitution":{"type":"number","minimum":0,"maximum":1},"shape":{"enum":["box","sphere","convex_hull","mesh"]},"linear_damping":{"type":"number","minimum":0},"angular_damping":{"type":"number","minimum":0},"initial_velocity":{"$ref":"#/$defs/vec3"}},"additionalProperties":false},
        "force_field":{"type":"object","properties":{"type":{"enum":["wind","vortex","force"]},"strength":{"type":"number"},"falloff":{"type":"number","minimum":0}},"additionalProperties":false},
        "collection":{"type":"object","required":["name","children","objects"],"properties":{"name":{"type":"string"},"children":{"type":"array","items":{"$ref":"#/$defs/id"}},"objects":{"type":"array","items":{"$ref":"#/$defs/id"}}},"additionalProperties":false},
        "transform":{"type":"object","required":["translation","rotation","scale","rotation_mode"],"properties":{"translation":{"$ref":"#/$defs/vec3"},"rotation":{"type":"array","items":{"type":"number"},"minItems":4,"maxItems":4},"scale":{"$ref":"#/$defs/vec3"},"rotation_mode":{"type":"string"}},"additionalProperties":false},
        "modifier":{"type":"object","required":["id","type","name","enabled","params"],"properties":{"id":{"$ref":"#/$defs/id"},"type":{"type":"string"},"name":{"type":"string"},"enabled":{"type":"boolean"},"params":{"type":"object"}},"additionalProperties":false},
        "library":{"type":"object","required":["name","kind","uri","resolved_path","resource","hash","status","linked_ids","overrides","items","source_project"],"properties":{"name":{"type":"string"},"kind":{"enum":["potter_project","blend"]},"uri":{"type":"string"},"resolved_path":{"type":"string"},"resource":{"oneOf":[{"$ref":"#/$defs/id"},{"type":"null"}]},"hash":{"type":"string"},"status":{"enum":["ok","missing","changed"]},"linked_ids":{"type":"object","propertyNames":{"enum":["nodes","data_blocks","materials","collections","actions","node_groups","worlds","resources"]},"additionalProperties":{"type":"array","items":{"$ref":"#/$defs/id"}}},"overrides":{"type":"array","items":{"$ref":"#/$defs/library_override"}},"items":{"type":"object","additionalProperties":{"type":"string"}},"source_project":{"type":["string","null"]}},"additionalProperties":false},
        "library_override":{"type":"object","required":["registry","id","reference_id","properties"],"properties":{"registry":{"type":"string"},"id":{"$ref":"#/$defs/id"},"reference_id":{"$ref":"#/$defs/id"},"properties":{"type":"array","items":{"$ref":"#/$defs/library_override_property"}}},"additionalProperties":false},
        "library_override_property":{"type":"object","required":["path","operation","value"],"properties":{"path":{"type":"string"},"operation":{"enum":["replace","insert_after","delete"]},"value":{}},"additionalProperties":false},
        "node":{"type":"object","required":["name","kind"],"properties":{"name":{"type":"string"},"kind":{"enum":["empty","mesh","camera","light"]},"primitive":{"type":["string","null"]},"tags":{"type":"array","items":{"$ref":"#/$defs/id"}},"parent":{"oneOf":[{"$ref":"#/$defs/id"},{"type":"null"}]},"parent_inverse":{"oneOf":[{"type":"array","items":{"type":"number"},"minItems":16,"maxItems":16},{"type":"null"}]},"transform":{"$ref":"#/$defs/transform"},"data":{"oneOf":[{"$ref":"#/$defs/id"},{"type":"null"}]},"materials":{"type":"array","items":{"$ref":"#/$defs/id"}},"modifiers":{"type":"array","items":{"$ref":"#/$defs/modifier"}},"visible":{"type":"boolean"},"render_visible":{"type":"boolean"},"selectable":{"type":"boolean"},"action":{"oneOf":[{"$ref":"#/$defs/id"},{"type":"null"}]},"nla_tracks":{"type":"array","items":{"type":"object"}},"properties":{"type":"object"},"rigid_body":{"oneOf":[{"$ref":"#/$defs/rigid_body"},{"type":"null"}]},"force_field":{"oneOf":[{"$ref":"#/$defs/force_field"},{"type":"null"}]},"parent_type":{"enum":["object","bone"]},"parent_bone":{"oneOf":[{"$ref":"#/$defs/id"},{"type":"null"}]},"pose":{"type":"object","additionalProperties":{"type":"object"}},"constraints":{"type":"array","items":{"type":"object"}},"drivers":{"type":"array","items":{"type":"object"}}},"additionalProperties":false},
        "vertex":{"type":"object","required":["id","co"],"properties":{"id":{"type":"integer","minimum":0,"maximum":u32::MAX},"co":{"$ref":"#/$defs/vec3"}},"additionalProperties":false},
        "edge":{"type":"object","required":["id","v"],"properties":{"id":{"type":"integer","minimum":0,"maximum":u32::MAX},"v":{"type":"array","items":{"type":"integer","minimum":0,"maximum":u32::MAX},"minItems":2,"maxItems":2}},"additionalProperties":false},
        "face":{"type":"object","required":["id","v","material_index"],"properties":{"id":{"type":"integer","minimum":0,"maximum":u32::MAX},"v":{"type":"array","items":{"type":"integer","minimum":0,"maximum":u32::MAX},"minItems":3},"material_index":{"type":"integer","minimum":0,"maximum":u32::MAX}},"additionalProperties":false},
        "mesh":{"type":"object","required":["vertices","edges","faces","attributes","next_id"],"properties":{"vertices":{"type":"array","items":{"$ref":"#/$defs/vertex"}},"edges":{"type":"array","items":{"$ref":"#/$defs/edge"}},"faces":{"type":"array","items":{"$ref":"#/$defs/face"}},"attributes":{"type":"object"},"next_id":{"type":"object","properties":{"vertex":{"type":"integer","minimum":0},"edge":{"type":"integer","minimum":0},"face":{"type":"integer","minimum":0}},"required":["vertex","edge","face"],"additionalProperties":false}},"additionalProperties":false},
        "data_block":{"type":"object","required":["type"],"properties":{"type":{"type":"string"},"descriptor":{"type":["object","null"]},"mesh":{"oneOf":[{"$ref":"#/$defs/mesh"},{"type":"null"}]},"camera":{"oneOf":[{"$ref":"#/$defs/camera"},{"type":"null"}]},"light":{"oneOf":[{"$ref":"#/$defs/light"},{"type":"null"}]},"grease_pencil":{"type":["object","null"]},"armature":{"type":["object","null"]},"shape_keys":{"type":["object","null"]},"vertex_groups":{"type":"array","items":{"type":"object"}},"vertex_weights":{"type":"object"}},"additionalProperties":false},
        "camera":{"type":"object","required":["projection","lens_mm","sensor_width_mm","ortho_scale","clip_start","clip_end","shift"],"properties":{"projection":{"enum":["perspective","orthographic"]},"lens_mm":{"type":"number"},"sensor_width_mm":{"type":"number"},"ortho_scale":{"type":"number"},"clip_start":{"type":"number"},"clip_end":{"type":"number"},"shift":{"$ref":"#/$defs/vec2"}},"additionalProperties":false},
        "light":{"type":"object","required":["light_type","color","energy","radius","spot_size","spot_blend"],"properties":{"light_type":{"enum":["point","sun","spot","area"]},"color":{"$ref":"#/$defs/vec3"},"energy":{"type":"number"},"radius":{"type":"number"},"spot_size":{"type":"number"},"spot_blend":{"type":"number"}},"additionalProperties":false},
        "material":{"type":"object","required":["name","base_color","metallic","roughness","emission_color","emission_strength","transmission","ior","double_sided","node_tree","base_color_texture","roughness_texture","metallic_texture","normal_texture"],"properties":{"name":{"type":"string"},"base_color":{"type":"array","items":{"type":"number","minimum":0,"maximum":1},"minItems":4,"maxItems":4},"metallic":{"type":"number","minimum":0,"maximum":1},"roughness":{"type":"number","minimum":0,"maximum":1},"emission_color":{"$ref":"#/$defs/vec3"},"emission_strength":{"type":"number","minimum":0},"transmission":{"type":"number","minimum":0,"maximum":1},"ior":{"type":"number","exclusiveMinimum":0},"double_sided":{"type":"boolean"},"node_tree":{"oneOf":[{"$ref":"#/$defs/id"},{"type":"null"}]},"base_color_texture":{"oneOf":[{"$ref":"#/$defs/texture_ref"},{"type":"null"}]},"roughness_texture":{"oneOf":[{"$ref":"#/$defs/texture_ref"},{"type":"null"}]},"metallic_texture":{"oneOf":[{"$ref":"#/$defs/texture_ref"},{"type":"null"}]},"normal_texture":{"oneOf":[{"$ref":"#/$defs/texture_ref"},{"type":"null"}]}},"additionalProperties":false},
        "texture_ref":{"type":"object","required":["image","uv_map","interpolation"],"properties":{"image":{"$ref":"#/$defs/id"},"uv_map":{"type":["string","null"]},"interpolation":{"enum":["linear","closest"]}},"additionalProperties":false},
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
    root
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
        "uv.pin",
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
                "emission_color",
                "emission_strength",
                "transmission",
                "ior",
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
            vec!["target", "id", "type", "name", "enabled", "params"],
            vec!["op", "target", "id", "type"],
            "object",
            "object modifier stack",
            json!({"name":"type","enabled":true,"params":{}}),
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
                "target", "scope", "brush", "samples", "falloff", "symmetry", "seed", "delta",
                "dyntopo",
            ],
            vec!["op", "target", "brush", "samples"],
            "mesh object and stroke samples",
            "shared|single_user for shared data",
            json!({"falloff":"smooth","symmetry":[],"seed":0}),
        ),
        "sculpt.dyntopo" => (
            vec![
                "target",
                "scope",
                "samples",
                "falloff",
                "symmetry",
                "seed",
                "edge_length",
            ],
            vec!["op", "target", "samples", "edge_length"],
            "mesh object and stroke samples",
            "shared|single_user for shared data",
            json!({"falloff":"smooth","symmetry":[],"seed":0}),
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
            vec!["target", "elements", "scope", "method"],
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
            ],
            vec!["op", "target"],
            "mesh object and faces",
            "shared|single_user for shared data",
            json!({"translation":[0,0],"scale":[1,1],"rotation":0}),
        ),
        "uv.pin" => (
            vec!["target", "elements", "scope", "pinned"],
            vec!["op", "target"],
            "mesh object and faces",
            "shared|single_user for shared data",
            json!({"pinned":true}),
        ),
        "uv.pack" => (
            vec!["target", "elements", "scope", "margin"],
            vec!["op", "target"],
            "mesh object and faces",
            "shared|single_user for shared data",
            json!({"margin":0.04}),
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
        return json!({"type":"object","propertyNames":{"pattern":"^f[0-9]+$"},"additionalProperties":{"type":"array","items":vec_schema(3)}});
    }
    if operation.starts_with("sculpt.") {
        return match field {
            "scope" => json!({"enum":["shared","single_user"]}),
            "brush" => {
                json!({"enum":["draw","clay_strips","inflate","grab","smooth","flatten","pinch","crease","layer","snake_hook","thumb","rotate","nudge","blob","scrape","fill","draw_sharp","elastic_deform","pose_lite","boundary_lite"]})
            }
            "falloff" => {
                json!({"enum":["smooth","sphere","root","sharp","linear","constant"],"default":"smooth"})
            }
            "samples" => {
                json!({"type":"array","minItems":1,"items":{"type":"object","required":["position","pressure","radius","strength","time"],"properties":{"position":vec_schema(3),"pressure":{"type":"number","minimum":0,"maximum":1},"radius":{"type":"number","minimum":0},"strength":{"type":"number"},"time":{"type":"number"}},"additionalProperties":false}})
            }
            "symmetry" => json!({"type":"array","items":{"enum":["x","y","z"]},"uniqueItems":true}),
            "seed" => json!({"type":"integer","minimum":0,"maximum":u64::MAX}),
            "delta" => vec_schema(3),
            "dyntopo" => {
                json!({"type":"object","required":["edge_length"],"properties":{"edge_length":{"type":"number","exclusiveMinimum":0}},"additionalProperties":false})
            }
            "edge_length" => json!({"type":"number","exclusiveMinimum":0}),
            _ => common_field_schema(field),
        };
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
            "margin" => json!({"type":"number","minimum":0,"maximum":0.5,"default":0.04}),
            _ => common_field_schema(field),
        };
    }
    if ["camera.create", "camera.update"].contains(&operation) {
        return match field {
            "projection" => json!({"enum":["perspective","orthographic","panorama","fisheye"]}),
            "panorama_type" => json!({"enum":["equirectangular","fisheye_equidistant"]}),
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
        "kind" => {
            json!({"enum":["group","empty","mesh","camera","light","box","sphere","uv_sphere","cylinder","plane","cone","torus","icosphere","circle","grid"]})
        }
        "type" if operation == "modifier.create" => {
            json!({"enum":params::type_names(ParameterFamily::Modifier)})
        }
        "type" if operation == "mesh.attribute_create" => {
            json!({"enum":["float","int","float2","float3","color","byte_color","bool","quaternion"]})
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
        | "recursive" | "cascade_data" | "unlink" | "keep_world" | "merge" | "individual" => {
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
        "materials" => json!({"type":"array","items":id_schema()}),
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
            json!({"type":"number","minimum":0,"maximum":1})
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
    let properties = match operation {
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
            "params":{"type":"object"},"materials":{"type":"array","items":id_schema()},
            "visible":{"type":"boolean"},"render_visible":{"type":"boolean"},"selectable":{"type":"boolean"},
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
            "envelope_weight":{"type":"number","minimum":0,"maximum":1},
            "head_radius":{"type":"number","minimum":0},
            "tail_radius":{"type":"number","minimum":0}
        }),
        "bone_collection.update" => json!({
            "name":{"type":"string"},
            "visible":{"type":"boolean"}
        }),
        "material.update" => material_set_properties(),
        "modifier.update" => {
            json!({"name":{"type":"string"},"enabled":{"type":"boolean"},"type":{"enum":params::type_names(ParameterFamily::Modifier)},"params":{"type":"object"}})
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
            "rate":{"type":"number","minimum":0,"maximum":1_000_000},
            "emission_rate":{"type":"number","minimum":0,"maximum":1_000_000},
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
            "child_nbr":{"type":"integer","minimum":0,"maximum":1_000_000},
            "child_radius":{"type":"number","minimum":0},
            "child_type":{"enum":["NONE","SIMPLE","INTERPOLATED"]},
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
        "emission_color":{"type":"array","items":{"type":"number","minimum":0},"minItems":3,"maxItems":3},
        "emission_strength":{"type":"number","minimum":0},
        "transmission":{"type":"number","minimum":0,"maximum":1},
        "ior":{"type":"number","exclusiveMinimum":0},
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
        "projection":{"enum":["perspective","orthographic","panorama","fisheye"]},
        "lens_mm":{"type":"number","exclusiveMinimum":0},
        "sensor_width_mm":{"type":"number","exclusiveMinimum":0},
        "ortho_scale":{"type":"number","exclusiveMinimum":0},
        "clip_start":{"type":"number","exclusiveMinimum":0},
        "clip_end":{"type":"number","exclusiveMinimum":0},
        "shift":vec_schema(2),
        "panorama_type":{"enum":["equirectangular","fisheye_equidistant"]},
        "dof_enabled":{"type":"boolean"},
        "focus_distance":{"type":"number","exclusiveMinimum":0},
        "f_stop":{"type":"number","exclusiveMinimum":0},
        "aperture_blades":{"oneOf":[{"const":0},{"type":"integer","minimum":3,"maximum":16}]},
        "stereo_mode":{"enum":["none","side_by_side","anaglyph"]},
        "interocular_distance":{"type":"number","exclusiveMinimum":0}
    })
}

fn light_set_properties() -> Value {
    json!({"light_type":{"enum":["point","sun","spot","area"]},"color":{"type":"array","items":{"type":"number","minimum":0},"minItems":3,"maxItems":3},"energy":{"type":"number","minimum":0},"radius":{"type":"number","minimum":0},"spot_size":{"type":"number","exclusiveMinimum":0,"maximum":std::f64::consts::TAU},"spot_blend":{"type":"number","minimum":0,"maximum":1}})
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

    use serde_json::Value;
}
