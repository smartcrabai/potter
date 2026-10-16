//! Operations that build a VRM-exportable avatar: a skinned, shape-keyed body with MToon, expressions,
//! and spring bones. Copied from the known-good fixture in `tests/vrm_mtoon_texture_transform.rs`
//! without its texture-transform variant, so VRM 1.0 and VRM 0.x exports both succeed.

use serde_json::{Value, json};

fn image_operations() -> Vec<Value> {
    [
        ("tex_albedo", "Albedo", [0.8, 0.4, 0.2, 1.0]),
        ("tex_shade", "Shade", [0.2, 0.3, 0.4, 1.0]),
        ("tex_outline", "Outline", [0.6, 0.7, 0.8, 1.0]),
        ("tex_rim", "Rim", [0.5, 0.6, 0.7, 1.0]),
        ("tex_uv_mask", "UV Mask", [0.7, 0.8, 0.9, 1.0]),
        ("thumbnail", "Avatar Thumbnail", [0.9, 0.8, 0.7, 1.0]),
    ]
    .into_iter()
    .map(|(id, name, fill_color)| {
        json!({
            "op": "image.create",
            "id": id,
            "name": name,
            "width": 1,
            "height": 1,
            "fill_color": fill_color,
        })
    })
    .collect()
}

fn humanoid() -> Value {
    json!({
        "hips": "hips",
        "spine": "spine",
        "chest": "spine_001",
        "upperChest": "spine_002",
        "neck": "neck",
        "head": "head",
        "leftEye": "eye_l",
        "rightEye": "eye_r",
        "leftShoulder": "shoulder_l",
        "rightShoulder": "shoulder_r",
        "leftUpperArm": "upper_arm_l",
        "leftLowerArm": "forearm_l",
        "leftHand": "hand_l",
        "rightUpperArm": "upper_arm_r",
        "rightLowerArm": "forearm_r",
        "rightHand": "hand_r",
        "leftUpperLeg": "thigh_l",
        "leftLowerLeg": "shin_l",
        "leftFoot": "foot_l",
        "leftToes": "toe_l",
        "rightUpperLeg": "thigh_r",
        "rightLowerLeg": "shin_r",
        "rightFoot": "foot_r",
        "rightToes": "toe_r",
        "leftThumbMetacarpal": "thumb_meta_l",
        "leftThumbProximal": "thumb_prox_l",
        "leftThumbDistal": "thumb_distal_l",
        "rightThumbMetacarpal": "thumb_meta_r",
        "rightThumbProximal": "thumb_prox_r",
        "rightThumbDistal": "thumb_distal_r",
    })
}

fn meta() -> Value {
    json!({
        "name": "Native VRM Avatar",
        "authors": ["Potter VRM Test"],
        "license_url": "https://vrm.dev/licenses/1.0/",
        "thumbnail": "thumbnail",
        "avatar_permission": "only_author",
        "allow_excessively_violent_usage": false,
        "allow_excessively_sexual_usage": false,
        "commercial_usage": "personal_non_profit",
        "allow_political_or_religious_usage": false,
        "allow_antisocial_or_hate_usage": false,
        "credit_notation": "required",
        "allow_redistribution": false,
        "modification": "prohibited",
        "legacy": {
            "title": "Legacy Fixture Avatar",
            "author": "Potter VRM Test",
            "reference": "https://example.test/reference",
            "allowed_user": "only_author",
            "violent_usage": "disallow",
            "sexual_usage": "disallow",
            "commercial_usage": "disallow",
            "license_name": "redistribution_prohibited",
            "other_permission_url": "https://example.test/permissions",
        },
    })
}

fn spring_bone() -> Value {
    json!({
        "colliders": [{
            "node": {"node": "arm", "bone": "hair_root"},
            "shape": {"type": "sphere", "offset": [0.1, 0.2, 0.3], "radius": 0.08},
        }],
        "collider_groups": [{"name": "Hair", "colliders": [0]}],
        "springs": [{
            "name": "Hair",
            "center": {"node": "arm", "bone": "head"},
            "joints": [
                {
                    "node": {"node": "arm", "bone": "hair_root"},
                    "hit_radius": 0.02,
                    "stiffness": 0.8,
                    "gravity_power": 1.1,
                    "gravity_dir": [1.0, 2.0, 3.0],
                    "drag_force": 0.2,
                },
                {
                    "node": {"node": "arm", "bone": "hair_tip"},
                    "hit_radius": 0.02,
                    "stiffness": 0.8,
                    "gravity_power": 1.1,
                    "gravity_dir": [1.0, 2.0, 3.0],
                    "drag_force": 0.2,
                },
            ],
            "collider_groups": [0],
        }],
    })
}

fn mtoon() -> Value {
    json!({
        "transparent_with_z_write": false,
        "render_queue_offset_number": 0,
        "shade_color_factor": [0.3, 0.35, 0.4],
        "shade_multiply_texture": {"image": "tex_shade"},
        "shading_shift_factor": -0.15,
        "shading_toony_factor": 0.7,
        "gi_equalization_factor": 0.2,
        "matcap_factor": [1.0, 1.0, 1.0],
        "matcap_texture": {"image": "tex_outline"},
        "parametric_rim_color_factor": [0.1, 0.2, 0.3],
        "rim_multiply_texture": {"image": "tex_rim"},
        "rim_lighting_mix_factor": 0.6,
        "parametric_rim_fresnel_power_factor": 3.0,
        "parametric_rim_lift_factor": 0.1,
        "outline_width_mode": "world_coordinates",
        "outline_width_factor": 0.01,
        "outline_width_multiply_texture": {"image": "tex_outline"},
        "outline_color_factor": [0.15, 0.2, 0.25],
        "outline_lighting_mix_factor": 0.5,
        "uv_animation_mask_texture": {"image": "tex_uv_mask"},
        "uv_animation_scroll_x_speed_factor": 0.2,
        "uv_animation_scroll_y_speed_factor": -0.1,
        "uv_animation_rotation_speed_factor": 0.0,
    })
}

fn avatar() -> Value {
    let mut expressions = json!({
        "happy": {
            "is_binary": false,
            "override_blink": "none",
            "override_mouth": "none",
            "override_look_at": "none",
            "morph_target_binds": [{"node": "body", "shape_key": "smile", "weight": 0.4}],
            "material_color_binds": [{
                "material": "skin_material",
                "color_type": "color",
                "target_value": [0.2, 0.4, 0.6, 0.8],
            }],
        },
        "winkExtra": {
            "is_binary": true,
            "override_blink": "none",
            "override_mouth": "none",
            "override_look_at": "none",
            "morph_target_binds": [{"node": "body", "shape_key": "smile", "weight": 0.3}],
        },
    });
    for (preset, shape_key) in [
        ("lookLeft", "look_left"),
        ("lookRight", "look_right"),
        ("lookUp", "look_up"),
        ("lookDown", "look_down"),
    ] {
        expressions[preset] = json!({
            "is_binary": false,
            "morph_target_binds": [{"node": "body", "shape_key": shape_key, "weight": 0.65}],
        });
    }
    json!({
        "armature": "arm",
        "humanoid": humanoid(),
        "meta": meta(),
        "expressions": expressions,
        "first_person": {
            "mesh_annotations": [{"node": "body", "annotation_type": "first_person_only"}],
        },
        "look_at": {
            "look_at_type": "bone",
            "offset_from_head_bone": [0.1, 0.2, 0.3],
            "range_map_horizontal_inner": {"input_max_value": 30.0, "output_scale": 0.25},
            "range_map_horizontal_outer": {"input_max_value": 40.0, "output_scale": 0.5},
            "range_map_vertical_down": {"input_max_value": 50.0, "output_scale": 0.75},
            "range_map_vertical_up": {"input_max_value": 60.0, "output_scale": 1.0},
        },
        "spring_bone": spring_bone(),
        "mtoon": {"skin_material": mtoon()},
    })
}

/// Operations that create the avatar scene. The body is skinned to `arm` and carries shape keys.
pub fn avatar_operations() -> Vec<Value> {
    let mut operations = vec![
        // Shifts node indices away from mesh indices so a node/mesh index mix-up is detectable.
        json!({"op": "node.create", "id": "anchor", "kind": "empty"}),
        json!({"op": "node.create", "id": "arm", "kind": "armature"}),
        json!({"op": "rig.generate_basic_human", "target": {"id": "arm"}, "scale": 1.0}),
    ];
    operations.extend(image_operations());
    operations.extend([
        json!({
            "op": "material.create",
            "id": "skin_material",
            "name": "Avatar Skin",
            "base_color": [0.8, 0.65, 0.5, 1.0],
            "ior": 1.45,
            "base_color_texture": {"image": "tex_albedo", "uv_map": "uv_map"},
        }),
        json!({"op": "node.create", "id": "body", "kind": "box", "params": {"size": 1.0}, "material": "skin_material", "transform": {"scale": [1.0, 2.0, 0.5]}}),
        json!({"op": "uv.unwrap", "target": {"id": "body"}, "elements": {"domain": "face", "ids": ["f1"]}, "method": "smart"}),
        json!({"op": "vertex_group.create", "target": {"id": "body"}, "id": "hips", "name": "hips"}),
        json!({
            "op": "vertex_group.assign",
            "target": {"id": "body"},
            "group_id": "hips",
            "weights": (0..8).map(|vertex_id| json!({"vertex_id": vertex_id, "weight": 1.0})).collect::<Vec<_>>(),
        }),
        json!({"op": "modifier.create", "target": {"id": "body"}, "id": "skin_binding", "type": "armature", "params": {"object": "arm", "use_vertex_groups": true}}),
        json!({"op": "shape_key.create", "target": {"id": "body"}, "id": "smile", "name": "Smile", "positions": {"0": [2.5, 3.5, 4.5]}}),
        json!({"op": "shape_key.create", "target": {"id": "body"}, "id": "look_left", "name": "Look Left", "positions": {"1": [2.6, 3.5, 4.5]}}),
        json!({"op": "shape_key.create", "target": {"id": "body"}, "id": "look_right", "name": "Look Right", "positions": {"2": [2.5, 3.6, 4.5]}}),
        json!({"op": "shape_key.create", "target": {"id": "body"}, "id": "look_up", "name": "Look Up", "positions": {"3": [2.5, 3.5, 4.6]}}),
        json!({"op": "shape_key.create", "target": {"id": "body"}, "id": "look_down", "name": "Look Down", "positions": {"4": [2.5, 3.5, 4.4]}}),
        json!({"op": "shape_key.update", "target": {"id": "body"}, "id": "smile", "set": {"value": 0.0}}),
    ]);
    operations.extend([
        json!({"op": "bone.create", "target": {"id": "arm"}, "id": "thumb_meta_l", "name": "thumb_metacarpal.L", "head": [0.2, 0.0, 1.5], "tail": [0.3, 0.0, 1.52], "parent": "hand_l"}),
        json!({"op": "bone.create", "target": {"id": "arm"}, "id": "thumb_prox_l", "name": "thumb_proximal.L", "head": [0.3, 0.0, 1.52], "tail": [0.4, 0.0, 1.54], "parent": "thumb_meta_l"}),
        json!({"op": "bone.create", "target": {"id": "arm"}, "id": "thumb_distal_l", "name": "thumb_distal.L", "head": [0.4, 0.0, 1.54], "tail": [0.5, 0.0, 1.56], "parent": "thumb_prox_l"}),
        json!({"op": "bone.create", "target": {"id": "arm"}, "id": "thumb_meta_r", "name": "thumb_metacarpal.R", "head": [-0.2, 0.0, 1.5], "tail": [-0.3, 0.0, 1.52], "parent": "hand_r"}),
        json!({"op": "bone.create", "target": {"id": "arm"}, "id": "thumb_prox_r", "name": "thumb_proximal.R", "head": [-0.3, 0.0, 1.52], "tail": [-0.4, 0.0, 1.54], "parent": "thumb_meta_r"}),
        json!({"op": "bone.create", "target": {"id": "arm"}, "id": "thumb_distal_r", "name": "thumb_distal.R", "head": [-0.4, 0.0, 1.54], "tail": [-0.5, 0.0, 1.56], "parent": "thumb_prox_r"}),
        json!({"op": "bone.create", "target": {"id": "arm"}, "id": "eye_l", "name": "Eye.L", "head": [0.04, 0.0, 1.78], "tail": [0.04, 0.03, 1.78], "parent": "head"}),
        json!({"op": "bone.create", "target": {"id": "arm"}, "id": "eye_r", "name": "Eye.R", "head": [-0.04, 0.0, 1.78], "tail": [-0.04, 0.03, 1.78], "parent": "head"}),
        // Tilted, like the harness hair chain: an axis-aligned rest frame has no f32 rounding in its axes.
        json!({"op": "bone.create", "target": {"id": "arm"}, "id": "hair_root", "name": "Hair Root", "head": [0.0, 0.0, 1.9], "tail": [0.03, 0.02, 2.05], "parent": "head"}),
        json!({"op": "bone.create", "target": {"id": "arm"}, "id": "hair_tip", "name": "Hair Tip", "head": [0.0, 0.0, 2.05], "tail": [0.0, 0.0, 2.2], "parent": "hair_root"}),
        json!({"op": "avatar.update", "set": avatar()}),
    ]);
    operations
}
