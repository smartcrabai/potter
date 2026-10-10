#![expect(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "integration test setup"
)]

#[path = "common/blender_file.rs"]
mod blender_file;

use std::process::Command;

use potter_core::{
    error::ErrorCode,
    graph::{self, GraphKind, GraphLink, GraphNode, NodeGroup},
    model::{Id, Material, Registry},
    shader::{HitContext, evaluate_surface},
};
use serde_json::{Map, Value, json};
use tempfile::tempdir;

#[derive(Clone)]
struct NodeCase {
    id: String,
    kind: &'static str,
    operation: &'static str,
    inputs: Value,
    output: &'static str,
    data_type: &'static str,
}

impl NodeCase {
    fn blender_input(&self) -> Value {
        let inputs = self.inputs.as_object().unwrap();
        let input_values: Vec<&Value> = if self.kind == "math" {
            vec![&inputs["Value"], &inputs["Value_001"], &inputs["Value_002"]]
        } else {
            vec![
                &inputs["Vector"],
                &inputs["Vector_001"],
                &inputs["Vector_002"],
                &inputs["Scale"],
            ]
        };
        json!({
            "id": self.id,
            "kind": self.kind,
            "operation": self.operation,
            "inputs": input_values,
            "output": self.output,
            "data_type": self.data_type,
        })
    }
}

fn math_case(id: &str, operation: &'static str, first: f64, second: f64, third: f64) -> NodeCase {
    NodeCase {
        id: id.to_owned(),
        kind: "math",
        operation,
        inputs: json!({
            "Value": first,
            "Value_001": second,
            "Value_002": third,
        }),
        output: "Value",
        data_type: "FLOAT",
    }
}

fn vector_case(
    id: &str,
    operation: &'static str,
    first: [f64; 3],
    second: [f64; 3],
    third: [f64; 3],
    scale: f64,
    output: &'static str,
) -> NodeCase {
    NodeCase {
        id: id.to_owned(),
        kind: "vector",
        operation,
        inputs: json!({
            "Vector": first,
            "Vector_001": second,
            "Vector_002": third,
            "Scale": scale,
        }),
        output,
        data_type: if output == "Value" {
            "FLOAT"
        } else {
            "FLOAT_VECTOR"
        },
    }
}

fn math_cases() -> Vec<NodeCase> {
    let pi = std::f64::consts::PI;
    vec![
        math_case("math_add", "ADD", 3.25, 2.0, 0.0),
        math_case("math_subtract", "SUBTRACT", 3.25, 2.0, 0.0),
        math_case("math_multiply", "MULTIPLY", -2.5, 3.0, 0.0),
        math_case("math_multiply_add", "MULTIPLY_ADD", 2.0, 3.0, 4.0),
        math_case("math_divide", "DIVIDE", 7.5, 2.5, 0.0),
        math_case("math_power", "POWER", 2.0, 3.0, 0.0),
        math_case("math_logarithm", "LOGARITHM", 8.0, 2.0, 0.0),
        math_case("math_sqrt", "SQRT", 4.0, 0.0, 0.0),
        math_case("math_inverse_sqrt", "INVERSE_SQRT", 4.0, 0.0, 0.0),
        math_case("math_absolute", "ABSOLUTE", -2.5, 0.0, 0.0),
        math_case("math_exponent", "EXPONENT", 1.0, 0.0, 0.0),
        math_case("math_minimum", "MINIMUM", 2.0, 3.0, 0.0),
        math_case("math_maximum", "MAXIMUM", 2.0, 3.0, 0.0),
        math_case("math_less_than", "LESS_THAN", 2.0, 3.0, 0.0),
        math_case("math_greater_than", "GREATER_THAN", 3.0, 2.0, 0.0),
        math_case("math_sign", "SIGN", -2.5, 0.0, 0.0),
        math_case("math_compare_within_epsilon", "COMPARE", 3.0, 3.0005, 0.001),
        math_case(
            "math_compare_outside_epsilon",
            "COMPARE",
            3.0,
            3.0005,
            0.0001,
        ),
        math_case("math_smooth_min", "SMOOTH_MIN", 0.25, 0.75, 1.0),
        math_case("math_round_half_positive", "ROUND", 2.5, 0.0, 0.0),
        math_case("math_round_half_negative", "ROUND", -2.5, 0.0, 0.0),
        math_case("math_snap_rounds_down", "SNAP", 2.8, 0.5, 0.0),
        math_case("math_snap_negative", "SNAP", -2.8, 0.5, 0.0),
        math_case("math_snap_negative_increment", "SNAP", 2.8, -0.5, 0.0),
        math_case(
            "math_power_negative_integer_exponent",
            "POWER",
            -2.0,
            3.0,
            0.0,
        ),
        math_case("math_power_negative_even_exponent", "POWER", -2.0, 2.0, 0.0),
        math_case("math_power_zero_zero", "POWER", 0.0, 0.0, 0.0),
        math_case("math_power_zero_negative", "POWER", 0.0, -1.0, 0.0),
        math_case("math_log_base_one", "LOGARITHM", 8.0, 1.0, 0.0),
        math_case("math_log_negative_base", "LOGARITHM", 8.0, -2.0, 0.0),
        math_case("math_asin_out_of_range", "ARCSINE", 2.0, 0.0, 0.0),
        math_case("math_acos_out_of_range", "ARCCOSINE", -2.0, 0.0, 0.0),
        math_case("math_arctan2_origin", "ARCTAN2", 0.0, 0.0, 0.0),
        math_case("math_sign_zero", "SIGN", 0.0, 0.0, 0.0),
        math_case("math_modulo_negative_divisor", "MODULO", 5.5, -2.0, 0.0),
        math_case(
            "math_floored_modulo_negative_dividend",
            "FLOORED_MODULO",
            -5.5,
            2.0,
            0.0,
        ),
        math_case("math_fract_positive", "FRACT", 2.75, 0.0, 0.0),
        math_case("math_wrap_below_range", "WRAP", -7.5, 1.0, 4.0),
        math_case("math_pingpong_negative", "PINGPONG", -1.5, 2.0, 0.0),
        math_case("math_pingpong_negative_scale", "PINGPONG", 3.0, -2.0, 0.0),
        math_case("math_smooth_min_far_apart", "SMOOTH_MIN", 0.0, 5.0, 1.0),
        math_case("math_smooth_min_equal", "SMOOTH_MIN", 1.0, 1.0, 0.5),
        math_case("math_smooth_max_equal", "SMOOTH_MAX", 1.0, 1.0, 0.5),
        math_case(
            "math_compare_negative_epsilon",
            "COMPARE",
            1.0,
            1.000_005,
            -1.0,
        ),
        math_case(
            "math_compare_below_minimum_epsilon",
            "COMPARE",
            1.0,
            1.001,
            0.0,
        ),
        math_case("math_less_than_equal", "LESS_THAN", 2.0, 2.0, 0.0),
        math_case("math_greater_than_equal", "GREATER_THAN", 2.0, 2.0, 0.0),
        math_case("math_minimum_negative", "MINIMUM", -2.0, -3.0, 0.0),
        math_case("math_trunc_positive", "TRUNC", 2.7, 0.0, 0.0),
        math_case("math_ceil_positive", "CEIL", 2.1, 0.0, 0.0),
        math_case("math_exponent_negative", "EXPONENT", -2.0, 0.0, 0.0),
        math_case("math_inverse_sqrt_zero", "INVERSE_SQRT", 0.0, 0.0, 0.0),
        math_case("math_divide_negative", "DIVIDE", -7.5, 2.5, 0.0),
        math_case("math_smooth_max", "SMOOTH_MAX", 0.25, 0.75, 1.0),
        math_case("math_smooth_min_zero_width", "SMOOTH_MIN", 0.25, 0.75, 0.0),
        math_case(
            "math_smooth_max_negative_width",
            "SMOOTH_MAX",
            0.25,
            0.75,
            -1.0,
        ),
        math_case("math_round", "ROUND", 2.6, 0.0, 0.0),
        math_case("math_floor", "FLOOR", -2.3, 0.0, 0.0),
        math_case("math_ceil", "CEIL", -2.3, 0.0, 0.0),
        math_case("math_trunc", "TRUNC", -2.7, 0.0, 0.0),
        math_case("math_fract_negative", "FRACT", -2.25, 0.0, 0.0),
        math_case("math_modulo_negative", "MODULO", -5.5, 2.0, 0.0),
        math_case(
            "math_floored_modulo_negative_divisor",
            "FLOORED_MODULO",
            5.5,
            -2.0,
            0.0,
        ),
        math_case("math_wrap", "WRAP", 7.5, 1.0, 4.0),
        math_case("math_wrap_reversed_bounds", "WRAP", -2.5, 4.0, 1.0),
        math_case("math_snap", "SNAP", 2.7, 0.5, 0.0),
        math_case("math_pingpong", "PINGPONG", 5.5, 2.0, 0.0),
        math_case("math_sine", "SINE", 0.5, 0.0, 0.0),
        math_case("math_cosine", "COSINE", 0.5, 0.0, 0.0),
        math_case("math_tangent", "TANGENT", 0.5, 0.0, 0.0),
        math_case("math_arcsine", "ARCSINE", 0.5, 0.0, 0.0),
        math_case("math_arccosine", "ARCCOSINE", 0.5, 0.0, 0.0),
        math_case("math_arctangent", "ARCTANGENT", 0.5, 0.0, 0.0),
        math_case("math_arctan2", "ARCTAN2", 1.0, 2.0, 0.0),
        math_case("math_sinh", "SINH", 0.5, 0.0, 0.0),
        math_case("math_cosh", "COSH", 0.5, 0.0, 0.0),
        math_case("math_tanh", "TANH", 0.5, 0.0, 0.0),
        math_case("math_degrees", "DEGREES", pi, 0.0, 0.0),
        math_case("math_radians", "RADIANS", 180.0, 0.0, 0.0),
        math_case("math_divide_by_zero", "DIVIDE", 5.0, 0.0, 0.0),
        math_case("math_negative_power_domain", "POWER", -4.0, 0.5, 0.0),
        math_case("math_negative_log_domain", "LOGARITHM", -4.0, 2.0, 0.0),
        math_case("math_zero_log_domain", "LOGARITHM", 0.0, 2.0, 0.0),
        math_case("math_negative_sqrt_domain", "SQRT", -4.0, 0.0, 0.0),
        math_case(
            "math_negative_inverse_sqrt_domain",
            "INVERSE_SQRT",
            -4.0,
            0.0,
            0.0,
        ),
        math_case("math_modulo_by_zero", "MODULO", 5.0, 0.0, 0.0),
        math_case(
            "math_floored_modulo_by_zero",
            "FLOORED_MODULO",
            5.0,
            0.0,
            0.0,
        ),
        math_case("math_wrap_equal_bounds", "WRAP", 5.0, 2.0, 2.0),
        math_case("math_snap_zero_increment", "SNAP", 5.0, 0.0, 0.0),
        math_case("math_pingpong_zero_scale", "PINGPONG", 5.0, 0.0, 0.0),
    ]
}

fn vector_cases() -> Vec<NodeCase> {
    let first = [1.25, -2.5, 3.75];
    let second = [2.0, 4.0, 0.5];
    let third = [-1.0, 0.25, 3.0];
    vec![
        vector_case("vector_add", "ADD", first, second, third, 1.5, "Vector"),
        vector_case(
            "vector_subtract",
            "SUBTRACT",
            first,
            second,
            third,
            1.5,
            "Vector",
        ),
        vector_case(
            "vector_multiply",
            "MULTIPLY",
            first,
            second,
            third,
            1.5,
            "Vector",
        ),
        vector_case("vector_power", "POWER", first, second, third, 1.5, "Vector"),
        vector_case(
            "vector_divide",
            "DIVIDE",
            first,
            second,
            third,
            1.5,
            "Vector",
        ),
        vector_case(
            "vector_cross_product",
            "CROSS_PRODUCT",
            first,
            second,
            third,
            1.5,
            "Vector",
        ),
        vector_case(
            "vector_project",
            "PROJECT",
            first,
            second,
            third,
            1.5,
            "Vector",
        ),
        vector_case(
            "vector_reflect",
            "REFLECT",
            first,
            second,
            third,
            1.5,
            "Vector",
        ),
        vector_case(
            "vector_refract",
            "REFRACT",
            [0.25, -0.5, -0.8],
            [0.0, 0.0, 1.0],
            third,
            1.2,
            "Vector",
        ),
        vector_case(
            "vector_faceforward",
            "FACEFORWARD",
            first,
            second,
            third,
            1.5,
            "Vector",
        ),
        vector_case(
            "vector_dot_product",
            "DOT_PRODUCT",
            first,
            second,
            third,
            1.5,
            "Value",
        ),
        vector_case(
            "vector_distance",
            "DISTANCE",
            first,
            second,
            third,
            1.5,
            "Value",
        ),
        vector_case(
            "vector_length",
            "LENGTH",
            first,
            second,
            third,
            1.5,
            "Value",
        ),
        vector_case("vector_scale", "SCALE", first, second, third, 1.5, "Vector"),
        vector_case(
            "vector_normalize",
            "NORMALIZE",
            first,
            second,
            third,
            1.5,
            "Vector",
        ),
        vector_case(
            "vector_absolute",
            "ABSOLUTE",
            first,
            second,
            third,
            1.5,
            "Vector",
        ),
        vector_case(
            "vector_minimum",
            "MINIMUM",
            first,
            second,
            third,
            1.5,
            "Vector",
        ),
        vector_case(
            "vector_maximum",
            "MAXIMUM",
            first,
            second,
            third,
            1.5,
            "Vector",
        ),
        vector_case("vector_floor", "FLOOR", first, second, third, 1.5, "Vector"),
        vector_case("vector_ceil", "CEIL", first, second, third, 1.5, "Vector"),
        vector_case(
            "vector_round",
            "ROUND",
            [1.5, -2.5, 3.5],
            second,
            third,
            1.5,
            "Vector",
        ),
        vector_case(
            "vector_fraction",
            "FRACTION",
            first,
            second,
            third,
            1.5,
            "Vector",
        ),
        vector_case(
            "vector_modulo",
            "MODULO",
            first,
            second,
            third,
            1.5,
            "Vector",
        ),
        vector_case(
            "vector_snap",
            "SNAP",
            [2.7, -3.5, 7.1],
            [0.5, 1.5, 2.0],
            third,
            1.5,
            "Vector",
        ),
        vector_case(
            "vector_wrap",
            "WRAP",
            [5.2, -2.5, 3.7],
            [0.0, 1.0, -2.0],
            [2.0, 4.0, 2.0],
            1.5,
            "Vector",
        ),
        vector_case("vector_sine", "SINE", first, second, third, 1.5, "Vector"),
        vector_case(
            "vector_cosine",
            "COSINE",
            first,
            second,
            third,
            1.5,
            "Vector",
        ),
        vector_case(
            "vector_tangent",
            "TANGENT",
            first,
            second,
            third,
            1.5,
            "Vector",
        ),
        vector_case("vector_sign", "SIGN", first, second, third, 1.5, "Vector"),
        vector_case(
            "vector_multiply_add",
            "MULTIPLY_ADD",
            first,
            second,
            third,
            1.5,
            "Vector",
        ),
        vector_case(
            "vector_project_zero",
            "PROJECT",
            first,
            [0.0; 3],
            third,
            1.5,
            "Vector",
        ),
        vector_case(
            "vector_normalize_zero",
            "NORMALIZE",
            [0.0; 3],
            second,
            third,
            1.5,
            "Vector",
        ),
        vector_case(
            "vector_divide_by_zero",
            "DIVIDE",
            first,
            [1.0, 0.0, 2.0],
            third,
            1.5,
            "Vector",
        ),
        vector_case(
            "vector_modulo_by_zero",
            "MODULO",
            first,
            [1.0, 0.0, 2.0],
            third,
            1.5,
            "Vector",
        ),
        vector_case(
            "vector_snap_zero_increment",
            "SNAP",
            first,
            [1.0, 0.0, 2.0],
            third,
            1.5,
            "Vector",
        ),
        vector_case(
            "vector_wrap_equal_bounds",
            "WRAP",
            first,
            [1.0, 2.0, 3.0],
            [1.0, 2.0, 3.0],
            1.5,
            "Vector",
        ),
        vector_case(
            "vector_power_negative_domain",
            "POWER",
            [-4.0, 4.0, 9.0],
            [0.5, 0.5, 0.5],
            third,
            1.5,
            "Vector",
        ),
        vector_case(
            "vector_refract_total_internal_reflection",
            "REFRACT",
            [0.99, 0.0, -0.1],
            [0.0, 0.0, 1.0],
            third,
            1.5,
            "Vector",
        ),
        vector_case(
            "vector_round_half",
            "ROUND",
            [0.5, -0.5, 2.5],
            second,
            third,
            1.5,
            "Vector",
        ),
        vector_case(
            "vector_snap_negative",
            "SNAP",
            [-2.8, 2.8, -0.2],
            [0.5, -0.5, 1.0],
            third,
            1.5,
            "Vector",
        ),
        vector_case(
            "vector_modulo_negative",
            "MODULO",
            [-5.5, 5.5, -5.5],
            [2.0, -2.0, -2.0],
            third,
            1.5,
            "Vector",
        ),
        vector_case(
            "vector_fraction_negative",
            "FRACTION",
            [-2.25, 2.75, -0.5],
            second,
            third,
            1.5,
            "Vector",
        ),
        vector_case(
            "vector_floor_negative",
            "FLOOR",
            [-2.25, 2.75, -0.5],
            second,
            third,
            1.5,
            "Vector",
        ),
        vector_case(
            "vector_ceil_negative",
            "CEIL",
            [-2.25, 2.75, -0.5],
            second,
            third,
            1.5,
            "Vector",
        ),
        vector_case(
            "vector_power_negative_integer",
            "POWER",
            [-2.0, -2.0, 0.0],
            [3.0, 2.0, 0.0],
            third,
            1.5,
            "Vector",
        ),
        vector_case(
            "vector_wrap_reversed_bounds",
            "WRAP",
            [5.2, -2.5, 3.7],
            [2.0, 4.0, 2.0],
            [0.0, 1.0, -2.0],
            1.5,
            "Vector",
        ),
        vector_case(
            "vector_project_onto_axis",
            "PROJECT",
            [1.0, 2.0, 3.0],
            [0.0, 0.0, 2.0],
            third,
            1.5,
            "Vector",
        ),
        vector_case(
            "vector_reflect_unnormalised",
            "REFLECT",
            [1.0, -1.0, 0.0],
            [0.0, 3.0, 0.0],
            third,
            1.5,
            "Vector",
        ),
        vector_case(
            "vector_faceforward_opposed",
            "FACEFORWARD",
            [1.0, 2.0, 3.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            1.5,
            "Vector",
        ),
        vector_case(
            "vector_faceforward_aligned",
            "FACEFORWARD",
            [1.0, 2.0, 3.0],
            [0.0, 0.0, -1.0],
            [0.0, 0.0, 1.0],
            1.5,
            "Vector",
        ),
        vector_case(
            "vector_refract_zero_normal",
            "REFRACT",
            [0.25, -0.5, -0.8],
            [0.0; 3],
            third,
            1.2,
            "Vector",
        ),
        vector_case(
            "vector_pythagorean_length",
            "LENGTH",
            [3.0, 4.0, 12.0],
            second,
            third,
            1.5,
            "Value",
        ),
        vector_case(
            "vector_sign_zero",
            "SIGN",
            [0.0, -0.0, 5.0],
            second,
            third,
            1.5,
            "Vector",
        ),
        vector_case(
            "vector_divide_negative",
            "DIVIDE",
            first,
            [-2.0, 4.0, 0.25],
            third,
            1.5,
            "Vector",
        ),
    ]
}

fn add_node(group: &mut NodeGroup, id: &str, node_type: &str, inputs: Value, properties: Value) {
    let mut node = GraphNode::new(node_type);
    let Value::Object(inputs) = inputs else {
        panic!("node inputs must be an object");
    };
    node.inputs = inputs.into_iter().collect();
    let Value::Object(properties) = properties else {
        panic!("node properties must be an object");
    };
    node.properties = properties;
    group.nodes.insert(Id::new(id).unwrap(), node);
}

fn link(group: &mut NodeGroup, from: &str, from_socket: &str, to: &str, to_socket: &str) {
    group.links.push(GraphLink {
        from_node: Id::new(from).unwrap(),
        from_socket: from_socket.to_owned(),
        to_node: Id::new(to).unwrap(),
        to_socket: to_socket.to_owned(),
    });
}

fn evaluate_geometry_case(case: &NodeCase) -> potter_core::error::Result<Value> {
    let mut group = NodeGroup::new("Math parity", GraphKind::Geometry);
    let is_math = case.kind == "math";
    add_node(
        &mut group,
        "line",
        "GeometryNodeMeshLine",
        json!({"Count": 1}),
        json!({}),
    );
    add_node(
        &mut group,
        "operation",
        if is_math {
            "ShaderNodeMath"
        } else {
            "ShaderNodeVectorMath"
        },
        case.inputs.clone(),
        json!({"operation": case.operation}),
    );
    add_node(
        &mut group,
        "store",
        "GeometryNodeStoreNamedAttribute",
        json!({"Name": case.id}),
        json!({"domain": "POINT", "data_type": case.data_type}),
    );
    add_node(
        &mut group,
        "output",
        "NodeGroupOutput",
        json!({}),
        json!({}),
    );
    link(&mut group, "line", "Mesh", "store", "Geometry");
    link(&mut group, "operation", case.output, "store", "Value");
    link(&mut group, "store", "Geometry", "output", "Geometry");

    let evaluation = graph::evaluate(&group, None, &Map::new())?;
    Ok(evaluation.mesh.attributes[&case.id]["data"][0].clone())
}

fn evaluate_shader_case(case: &NodeCase) -> potter_core::error::Result<Value> {
    let mut group = NodeGroup::new("Shader math parity", GraphKind::Shader);
    let node_type = if case.kind == "math" {
        "ShaderNodeMath"
    } else {
        "ShaderNodeVectorMath"
    };
    add_node(
        &mut group,
        "operation",
        node_type,
        case.inputs.clone(),
        json!({"operation": case.operation}),
    );
    add_node(
        &mut group,
        "principled",
        "ShaderNodeBsdfPrincipled",
        json!({}),
        json!({}),
    );
    add_node(&mut group, "output", "OutputMaterial", json!({}), json!({}));
    let target_socket = if case.output == "Value" {
        "Emission Strength"
    } else {
        "Emission Color"
    };
    link(
        &mut group,
        "operation",
        case.output,
        "principled",
        target_socket,
    );
    link(&mut group, "principled", "BSDF", "output", "Surface");

    let graph_id = Id::new("shader_math_parity").unwrap();
    let mut groups = Registry::new();
    groups.insert(graph_id.clone(), group);
    let material = Material {
        node_tree: Some(graph_id),
        ..Material::default()
    };
    let context = HitContext::new(&groups);
    let surface = evaluate_surface(&material, &context)?;
    if case.output == "Value" {
        Ok(json!(surface.emission_strength))
    } else {
        Ok(json!(surface.emission_color))
    }
}

fn blender_reference(cases: &[NodeCase]) -> Map<String, Value> {
    let blender = blender_file::blender_executable().expect("Blender is available for this test");
    let directory = tempdir().unwrap();
    let cases_path = directory.path().join("cases.json");
    let script_path = directory.path().join("node_math_oracle.py");
    let cases_json = Value::Array(cases.iter().map(NodeCase::blender_input).collect());
    std::fs::write(&cases_path, serde_json::to_vec(&cases_json).unwrap()).unwrap();
    std::fs::write(&script_path, BLENDER_ORACLE).unwrap();
    let output = Command::new(blender)
        .args([
            "--background",
            "--factory-startup",
            "--python-exit-code",
            "1",
            "--python",
        ])
        .arg(&script_path)
        .arg("--")
        .arg(&cases_path)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "Blender Geometry Nodes oracle failed:\n{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    let json_line = stdout
        .lines()
        .find_map(|line| line.strip_prefix("POTTER_NODE_MATH="))
        .expect("Blender oracle emitted a result line");
    serde_json::from_str(json_line).unwrap()
}

const BLENDER_ORACLE: &str = r#"
import bpy
import json
import math
import sys

with open(sys.argv[sys.argv.index("--") + 1], encoding="utf-8") as source:
    cases = json.load(source)

tree = bpy.data.node_groups.new("Potter Math Oracle", "GeometryNodeTree")
tree.interface.new_socket(name="Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
tree.interface.new_socket(name="Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
nodes = tree.nodes
links = tree.links
mesh_line = nodes.new("GeometryNodeMeshLine")
mesh_line.inputs["Count"].default_value = 1
geometry_output = mesh_line.outputs["Mesh"]

for case in cases:
    if case["kind"] == "math":
        operation = nodes.new("ShaderNodeMath")
        operation.operation = case["operation"]
        for index, value in enumerate(case["inputs"]):
            operation.inputs[index].default_value = value
    else:
        operation = nodes.new("ShaderNodeVectorMath")
        operation.operation = case["operation"]
        for index, value in enumerate(case["inputs"]):
            operation.inputs[index].default_value = value
    store = nodes.new("GeometryNodeStoreNamedAttribute")
    store.domain = "POINT"
    store.data_type = case["data_type"]
    store.inputs["Name"].default_value = case["id"]
    links.new(geometry_output, store.inputs["Geometry"])
    links.new(operation.outputs[case["output"]], store.inputs["Value"])
    geometry_output = store.outputs["Geometry"]

group_output = nodes.new("NodeGroupOutput")
links.new(geometry_output, group_output.inputs["Geometry"])
mesh = bpy.data.meshes.new("Potter Math Oracle Mesh")
mesh.from_pydata([(0.0, 0.0, 0.0)], [], [])
obj = bpy.data.objects.new("Potter Math Oracle", mesh)
bpy.context.collection.objects.link(obj)
modifier = obj.modifiers.new("Potter Math Oracle", "NODES")
modifier.node_group = tree
depsgraph = bpy.context.evaluated_depsgraph_get()
evaluated = obj.evaluated_get(depsgraph)
evaluated_mesh = evaluated.to_mesh()
results = {}
for case in cases:
    attribute = evaluated_mesh.attributes.get(case["id"])
    if attribute is None or len(attribute.data) != 1:
        raise RuntimeError("missing expected evaluated attribute: " + case["id"])
    item = attribute.data[0]
    raw = list(item.vector) if case["data_type"] == "FLOAT_VECTOR" else [item.value]
    finite = all(math.isfinite(component) for component in raw)
    if not finite:
        results[case["id"]] = "non-finite"
    elif case["data_type"] == "FLOAT_VECTOR":
        results[case["id"]] = raw
    else:
        results[case["id"]] = raw[0]
evaluated.to_mesh_clear()
print("POTTER_NODE_MATH=" + json.dumps(results, sort_keys=True))
"#;

fn json_mismatch(actual: &Value, expected: &Value, label: &str) -> Option<String> {
    match (actual, expected) {
        (Value::Number(actual), Value::Number(expected)) => {
            let actual = actual.as_f64().unwrap();
            let expected = expected.as_f64().unwrap();
            let tolerance = 2.0e-5 * expected.abs().max(1.0);
            ((actual - expected).abs() > tolerance).then(|| {
                format!("{label}: Blender {expected:?}, Potter {actual:?} (tolerance {tolerance})")
            })
        }
        (Value::Array(actual), Value::Array(expected)) => actual
            .iter()
            .zip(expected)
            .enumerate()
            .find_map(|(axis, (actual, expected))| {
                json_mismatch(actual, expected, &format!("{label}[{axis}]"))
            }),
        _ => Some(format!(
            "{label}: Blender and Potter returned different shapes"
        )),
    }
}

fn compare_cases(
    cases: &[NodeCase],
    expected: &Map<String, Value>,
    prefix: &str,
    evaluate: impl Fn(&NodeCase) -> potter_core::error::Result<Value>,
    include: impl Fn(&NodeCase) -> bool,
) {
    let mut mismatches = Vec::new();
    for case in cases.iter().filter(|case| include(case)) {
        let blender_value = &expected[&case.id];
        let label = format!("{prefix}{}", case.id);
        match (evaluate(case), blender_value.is_string()) {
            (Ok(actual), false) => mismatches.extend(json_mismatch(&actual, blender_value, &label)),
            (Ok(actual), true) => mismatches.push(format!(
                "{label}: Blender produced a non-finite value, Potter returned {actual}"
            )),
            (Err(error), false) => mismatches.push(format!(
                "{label}: Blender {blender_value}, Potter error {error}"
            )),
            (Err(_), true) => {}
        }
    }
    assert!(
        mismatches.is_empty(),
        "{} of {} cases diverge from Blender 5.2.2:\n{}",
        mismatches.len(),
        cases.len(),
        mismatches.join("\n")
    );
}

#[test]
fn geometry_node_math_and_vector_math_match_blender_522() {
    let Some(_) = blender_file::blender_executable() else {
        eprintln!("Skipping Blender node math parity: no Blender executable was found");
        return;
    };
    let cases: Vec<NodeCase> = math_cases().into_iter().chain(vector_cases()).collect();
    let expected = blender_reference(&cases);
    compare_cases(
        &cases,
        &expected,
        "geometry_",
        evaluate_geometry_case,
        |_| true,
    );
}

/// Operations the shader evaluator documents as supported; others must report
/// `UNSUPPORTED_FEATURE` instead of silently evaluating.
fn shader_supports(case: &NodeCase) -> bool {
    match case.kind {
        "math" => matches!(
            case.operation,
            "ADD"
                | "SUBTRACT"
                | "MULTIPLY"
                | "DIVIDE"
                | "MULTIPLY_ADD"
                | "POWER"
                | "SINE"
                | "COSINE"
                | "TANGENT"
                | "SQRT"
                | "ABSOLUTE"
                | "MINIMUM"
                | "MAXIMUM"
                | "LESS_THAN"
                | "GREATER_THAN"
                | "COMPARE"
                | "PINGPONG"
        ),
        _ => matches!(
            case.operation,
            "ADD"
                | "SUBTRACT"
                | "MULTIPLY"
                | "CROSS_PRODUCT"
                | "DOT_PRODUCT"
                | "LENGTH"
                | "DISTANCE"
                | "SCALE"
                | "NORMALIZE"
        ),
    }
}

#[test]
fn shader_math_and_vector_math_match_blender_geometry_node_results() {
    let Some(_) = blender_file::blender_executable() else {
        eprintln!("Skipping Blender shader math parity: no Blender executable was found");
        return;
    };
    let cases: Vec<NodeCase> = math_cases().into_iter().chain(vector_cases()).collect();
    let expected = blender_reference(&cases);
    compare_cases(
        &cases,
        &expected,
        "shader_",
        evaluate_shader_case,
        shader_supports,
    );
}

fn shader_surface_with_node(
    node_type: &str,
    inputs: Value,
    properties: Value,
    output: &str,
    target: &str,
) -> potter_core::error::Result<potter_core::shader::BsdfParams> {
    let mut group = NodeGroup::new("Shader node", GraphKind::Shader);
    add_node(&mut group, "node", node_type, inputs, properties);
    add_node(
        &mut group,
        "principled",
        "ShaderNodeBsdfPrincipled",
        json!({}),
        json!({}),
    );
    add_node(&mut group, "output", "OutputMaterial", json!({}), json!({}));
    link(&mut group, "node", output, "principled", target);
    link(&mut group, "principled", "BSDF", "output", "Surface");
    let graph_id = Id::new("shader_node_case").unwrap();
    let mut groups = Registry::new();
    groups.insert(graph_id.clone(), group);
    let material = Material {
        node_tree: Some(graph_id),
        ..Material::default()
    };
    evaluate_surface(&material, &HitContext::new(&groups))
}

#[test]
fn unsupported_graph_math_operations_report_their_feature_id() {
    let math =
        evaluate_geometry_case(&math_case("m", "NOT_AN_OPERATION", 1.0, 1.0, 1.0)).unwrap_err();
    assert_eq!(math.code, ErrorCode::UnsupportedFeature);
    assert_eq!(
        math.details["feature_id"],
        "graph.shader_math.NOT_AN_OPERATION"
    );
    let vector = evaluate_geometry_case(&vector_case(
        "v",
        "NOT_AN_OPERATION",
        [1.0; 3],
        [1.0; 3],
        [1.0; 3],
        1.0,
        "Vector",
    ))
    .unwrap_err();
    assert_eq!(vector.code, ErrorCode::UnsupportedFeature);
    assert_eq!(
        vector.details["feature_id"],
        "graph.shader_vector_math.NOT_AN_OPERATION"
    );
}

#[test]
fn graph_vector_math_rejects_the_wrong_result_socket_kind() {
    for (operation, output) in [
        ("DOT_PRODUCT", "Vector"),
        ("DISTANCE", "Vector"),
        ("LENGTH", "Vector"),
        ("ADD", "Value"),
        ("NORMALIZE", "Value"),
    ] {
        let case = vector_case(
            "socket", operation, [1.0; 3], [2.0; 3], [3.0; 3], 1.0, output,
        );
        let error = evaluate_geometry_case(&case).unwrap_err();
        assert!(
            matches!(
                error.code,
                ErrorCode::InvalidOperation | ErrorCode::EvaluationFailed
            ),
            "{operation} -> {output}: {error}"
        );
    }
}

#[test]
fn graph_math_rejects_results_that_overflow_to_infinity() {
    for case in [
        math_case("exp", "EXPONENT", 1000.0, 0.0, 0.0),
        math_case("pow", "POWER", 0.0, -1.0, 0.0),
        math_case("mul", "MULTIPLY", 1.0e200, 1.0e200, 0.0),
        math_case("huge", "MULTIPLY", f64::MAX, 2.0, 0.0),
    ] {
        let error = evaluate_geometry_case(&case).unwrap_err();
        assert_eq!(error.code, ErrorCode::EvaluationFailed, "{}", case.id);
    }
    let vector = vector_case(
        "vmul",
        "MULTIPLY",
        [1.0e200; 3],
        [1.0e200; 3],
        [0.0; 3],
        1.0,
        "Vector",
    );
    assert_eq!(
        evaluate_geometry_case(&vector).unwrap_err().code,
        ErrorCode::EvaluationFailed
    );
    let scalar = vector_case(
        "vlen",
        "LENGTH",
        [1.0e200; 3],
        [0.0; 3],
        [0.0; 3],
        1.0,
        "Value",
    );
    assert_eq!(
        evaluate_geometry_case(&scalar).unwrap_err().code,
        ErrorCode::EvaluationFailed
    );
}

#[test]
fn shader_math_reports_unsupported_operations_and_wrong_sockets() {
    let unsupported = shader_surface_with_node(
        "ShaderNodeMath",
        json!({"Value": 1.0}),
        json!({"operation": "SIGN"}),
        "Value",
        "Emission Strength",
    )
    .unwrap_err();
    assert_eq!(unsupported.code, ErrorCode::UnsupportedFeature);
    assert_eq!(unsupported.details["feature_id"], "shader.node.Math:SIGN");

    let unsupported_vector = shader_surface_with_node(
        "ShaderNodeVectorMath",
        json!({"Vector": [1.0, 2.0, 3.0]}),
        json!({"operation": "REFLECT"}),
        "Vector",
        "Emission Color",
    )
    .unwrap_err();
    assert_eq!(unsupported_vector.code, ErrorCode::UnsupportedFeature);
    assert_eq!(
        unsupported_vector.details["feature_id"],
        "shader.node.VectorMath:REFLECT"
    );

    let wrong_socket = shader_surface_with_node(
        "ShaderNodeVectorMath",
        json!({"Vector": [1.0, 2.0, 3.0], "Vector_001": [1.0, 0.0, 0.0]}),
        json!({"operation": "ADD"}),
        "Value",
        "Emission Strength",
    )
    .unwrap_err();
    assert_eq!(wrong_socket.code, ErrorCode::InvalidOperation);

    let overflow = shader_surface_with_node(
        "ShaderNodeMath",
        json!({"Value": 1.0e200, "Value_001": 1.0e200}),
        json!({"operation": "MULTIPLY"}),
        "Value",
        "Emission Strength",
    )
    .unwrap_err();
    assert_eq!(overflow.code, ErrorCode::EvaluationFailed);
}

#[test]
fn shader_math_compare_defaults_to_blenders_node_epsilon() {
    let within = shader_surface_with_node(
        "ShaderNodeMath",
        json!({"Value": 1.0, "Value_001": 1.0005}),
        json!({"operation": "COMPARE"}),
        "Value",
        "Emission Strength",
    )
    .unwrap();
    assert!((within.emission_strength - 1.0).abs() < f64::EPSILON);
    let outside = shader_surface_with_node(
        "ShaderNodeMath",
        json!({"Value": 1.0, "Value_001": 1.002}),
        json!({"operation": "COMPARE"}),
        "Value",
        "Emission Strength",
    )
    .unwrap();
    assert!(outside.emission_strength.abs() < f64::EPSILON);
}

#[test]
fn voronoi_distance_is_bounded_deterministic_and_matches_its_feature_position() {
    for vector in [[0.3, 0.7, 1.2], [-4.2, 9.9, 0.0], [100.5, -0.25, 7.75]] {
        let inputs = json!({"Vector": vector, "Scale": 1.0});
        let distance = shader_surface_with_node(
            "ShaderNodeTexVoronoi",
            inputs.clone(),
            json!({}),
            "Distance",
            "Emission Strength",
        )
        .unwrap()
        .emission_strength;
        let repeated = shader_surface_with_node(
            "ShaderNodeTexVoronoi",
            inputs.clone(),
            json!({}),
            "Fac",
            "Emission Strength",
        )
        .unwrap()
        .emission_strength;
        assert_eq!(
            distance.to_bits(),
            repeated.to_bits(),
            "Fac aliases Distance"
        );
        // Each lattice cell holds one feature point, so the nearest point of the 27
        // neighbouring cells is never farther than the diagonal of one cell.
        assert!(
            (0.0..=3.0_f64.sqrt()).contains(&distance),
            "{vector:?}: {distance}"
        );
        let position = shader_surface_with_node(
            "ShaderNodeTexVoronoi",
            inputs,
            json!({}),
            "Position",
            "Emission Color",
        )
        .unwrap()
        .emission_color;
        let offset = [
            vector[0] - position[0],
            vector[1] - position[1],
            vector[2] - position[2],
        ];
        let measured = offset.iter().map(|axis| axis * axis).sum::<f64>().sqrt();
        assert!((measured - distance).abs() < 1.0e-12, "{vector:?}");
        for axis in 0..3 {
            assert!(
                (position[axis] - vector[axis].floor()).abs() < 2.0,
                "feature point stays within the neighbouring cells"
            );
        }
    }
}

#[test]
fn voronoi_scale_multiplies_the_sampled_coordinate_and_color_is_a_unit_rgb() {
    let sample = |scale: f64, socket: &str, target: &str| {
        shader_surface_with_node(
            "ShaderNodeTexVoronoi",
            json!({"Vector": [0.5, 1.0, 1.5], "Scale": scale}),
            json!({}),
            socket,
            target,
        )
        .unwrap()
    };
    let scaled = sample(2.0, "Position", "Emission Color").emission_color;
    let unit = shader_surface_with_node(
        "ShaderNodeTexVoronoi",
        json!({"Vector": [1.0, 2.0, 3.0], "Scale": 1.0}),
        json!({}),
        "Position",
        "Emission Color",
    )
    .unwrap()
    .emission_color;
    // The same scaled coordinate (1, 2, 3) must select the same feature point.
    assert_eq!(scaled.map(f64::to_bits), unit.map(f64::to_bits));
    let color = sample(2.0, "Color", "Emission Color").emission_color;
    assert!(color.iter().all(|channel| (0.0..=1.0).contains(channel)));
    let error = shader_surface_with_node(
        "ShaderNodeTexVoronoi",
        json!({}),
        json!({}),
        "Nope",
        "Emission Color",
    )
    .unwrap_err();
    assert_eq!(error.code, ErrorCode::InvalidOperation);
}
