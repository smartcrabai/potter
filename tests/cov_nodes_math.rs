#![expect(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "integration test setup"
)]

#[path = "common/blender_file.rs"]
mod blender_file;
#[path = "common/eevee_film.rs"]
mod eevee_film;
use eevee_film::assert_eevee_film_close;

use std::{
    collections::BTreeSet,
    path::{Path, PathBuf},
    process::Command,
};

use potter_core::{
    error::ErrorCode,
    graph::{self, GraphKind, GraphLink, GraphNode, NodeGroup},
    model::{Id, Material, Registry},
    shader::{HitContext, evaluate_surface},
};
use serde_json::{Map, Value, json};
use tempfile::{TempDir, tempdir};

#[derive(Clone)]
struct NodeCase {
    id: String,
    kind: &'static str,
    operation: &'static str,
    inputs: Value,
    output: &'static str,
    data_type: &'static str,
    clamp: bool,
    value_input: Option<f64>,
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
            "clamp": self.clamp,
            "value_input": self.value_input,
        })
    }

    fn properties(&self) -> Value {
        if self.kind == "math" {
            json!({"operation": self.operation, "use_clamp": self.clamp})
        } else {
            json!({"operation": self.operation})
        }
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
        clamp: false,
        value_input: None,
    }
}

fn clamped_math_case(
    id: &str,
    operation: &'static str,
    first: f64,
    second: f64,
    third: f64,
) -> NodeCase {
    NodeCase {
        clamp: true,
        ..math_case(id, operation, first, second, third)
    }
}

fn value_math_case() -> NodeCase {
    NodeCase {
        value_input: Some(1.5),
        ..math_case("math_shader_value_fold_probe", "ARCSINE", 0.0, 0.0, 0.0)
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
        clamp: false,
        value_input: None,
    }
}

fn math_cases() -> Vec<NodeCase> {
    let pi = std::f64::consts::PI;
    vec![
        value_math_case(),
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
        clamped_math_case("math_power_zero_negative_clamped", "POWER", 0.0, -1.0, 0.0),
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
        math_case("math_inverse_sqrt_zero", "INVERSE_SQRT", 0.0, 0.0, 0.0),
        math_case("math_divide_negative", "DIVIDE", -7.5, 2.5, 0.0),
        math_case("math_smooth_max", "SMOOTH_MAX", 0.25, 0.75, 1.0),
        math_case(
            "math_compare_zero_epsilon_small_difference",
            "COMPARE",
            0.0,
            1.0e-8,
            0.0,
        ),
        math_case(
            "math_compare_equal_infinities",
            "COMPARE",
            3.5e38,
            3.5e38,
            0.0,
        ),
        math_case("eevee_math_arcsine_in_domain", "ARCSINE", 0.25, 0.0, 0.0),
        math_case(
            "eevee_math_arccosine_in_domain",
            "ARCCOSINE",
            0.25,
            0.0,
            0.0,
        ),
        math_case("eevee_math_compare_defined", "COMPARE", 0.0005, 0.0, 0.001),
        math_case("eevee_math_power_integer", "POWER", -2.0, 2.0, 0.0),
        math_case(
            "eevee_math_modulo_safe_fractional",
            "MODULO",
            1.23,
            0.1,
            0.0,
        ),
        math_case(
            "math_compare_sub_half_float_epsilon",
            "COMPARE",
            0.000_005,
            0.0,
            0.0,
        ),
        math_case(
            "math_power_negative_near_integer",
            "POWER",
            -2.0,
            2.0005,
            0.0,
        ),
        math_case("math_modulo_precision_boundary", "MODULO", 1.0, 0.1, 0.0),
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
        math_case("math_exponent_overflow", "EXPONENT", 1000.0, 0.0, 0.0),
        clamped_math_case("math_clamp_above_one", "ADD", 0.75, 0.5, 0.0),
        clamped_math_case("math_clamp_below_zero", "SUBTRACT", 0.25, 0.5, 0.0),
        clamped_math_case("math_clamp_multiply_add", "MULTIPLY_ADD", 0.75, 0.75, 0.75),
        math_case(
            "math_sign_positive_infinite_input",
            "SIGN",
            3.5e38,
            0.0,
            0.0,
        ),
        math_case(
            "math_sign_negative_infinite_input",
            "SIGN",
            -3.5e38,
            0.0,
            0.0,
        ),
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
            "eevee_vector_sign_zero",
            "SIGN",
            [0.0, 2.0, 3.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            1.0,
            "Vector",
        ),
        vector_case(
            "eevee_vector_power_negative_near_integer",
            "POWER",
            [-2.0, -2.0, -2.0],
            [1.0005, 2.0, 0.0],
            [0.0, 0.0, 0.0],
            1.0,
            "Vector",
        ),
        vector_case(
            "eevee_vector_power_negative_integer",
            "POWER",
            [-0.5, -0.5, -0.5],
            [2.0, 2.0, 2.0],
            [0.0, 0.0, 0.0],
            1.0,
            "Vector",
        ),
        vector_case(
            "eevee_vector_refract_tiny_normal",
            "REFRACT",
            [1.0, 0.0, 0.0],
            [1.0e-20, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            1.0,
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
            "vector_refract_gn_incident_length",
            "REFRACT",
            [0.0, 0.0, 2.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0],
            1.0,
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
        vector_case(
            "vector_power_overflow",
            "POWER",
            [2.0, 2.0, 2.0],
            [2.0, 200.0, 3.0],
            third,
            1.5,
            "Vector",
        ),
        vector_case(
            "vector_sign_infinite_inputs",
            "SIGN",
            [3.5e38, -3.5e38, 0.0],
            [1.0, 1.0, 1.0],
            third,
            1.5,
            "Vector",
        ),
        vector_case(
            "eevee_vector_modulo_fractional_divisor",
            "MODULO",
            [1.0, 1.0, 1.0],
            [0.1, 0.1, 0.1],
            [0.0, 0.0, 0.0],
            1.0,
            "Vector",
        ),
        vector_case(
            "eevee_vector_reflect_tiny_normal",
            "REFLECT",
            [1.0, 0.0, 0.0],
            [1.0e-20, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            1.0,
            "Vector",
        ),
        vector_case(
            "eevee_vector_modulo_safe_fractional",
            "MODULO",
            [1.23, 0.7, 0.4],
            [0.1, 0.2, 0.3],
            [0.0, 0.0, 0.0],
            1.0,
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
        case.properties(),
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

    let evaluation = graph::evaluate(&group, None, &Map::new(), "Math parity")?;
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
        case.properties(),
    );
    if let Some(value) = case.value_input {
        add_node(
            &mut group,
            "value",
            "ShaderNodeValue",
            json!({}),
            json!({"potter_output_defaults": [{"identifier": "Value", "default": value}]}),
        );
        link(&mut group, "value", "Value", "operation", "Value");
    }
    add_node(
        &mut group,
        "emission",
        "ShaderNodeEmission",
        json!({"Color": [1.0, 1.0, 1.0, 1.0], "Strength": 1.0}),
        json!({}),
    );
    add_node(&mut group, "output", "OutputMaterial", json!({}), json!({}));
    let target_socket = if case.output == "Value" {
        "Strength"
    } else {
        "Color"
    };
    link(
        &mut group,
        "operation",
        case.output,
        "emission",
        target_socket,
    );
    link(&mut group, "emission", "Emission", "output", "Surface");

    let graph_id = Id::new("shader_math_parity").unwrap();
    let mut groups = Registry::new();
    groups.insert(graph_id.clone(), group);
    let material = Material {
        node_tree: Some(graph_id),
        ..Material::default()
    };
    let context = HitContext::new(&groups);
    let surface = evaluate_surface(&material, &context)?;
    let params = surface
        .dominant()
        .expect("an emission surface has one closure");
    if case.output == "Value" {
        Ok(json!(params.emission_strength))
    } else {
        Ok(json!(params.emission_color))
    }
}

struct BlenderOracle {
    directory: TempDir,
    geometry: Map<String, Value>,
    shader: Map<String, Value>,
    eevee: Map<String, Value>,
    shader_blend_path: PathBuf,
    eevee_blend_path: PathBuf,
}

fn is_eevee_probe(case: &NodeCase) -> bool {
    matches!(
        case.id.as_str(),
        "eevee_math_arcsine_in_domain"
            | "eevee_math_arccosine_in_domain"
            | "eevee_math_compare_defined"
            | "eevee_math_power_integer"
            | "eevee_math_modulo_safe_fractional"
            | "eevee_vector_sign_zero"
            | "vector_normalize_zero"
            | "eevee_vector_refract_tiny_normal"
            | "eevee_vector_reflect_tiny_normal"
            | "eevee_vector_power_negative_integer"
            | "eevee_vector_modulo_safe_fractional"
            | "math_shader_value_fold_probe"
    )
}

fn shader_cli_cases(cases: &[NodeCase]) -> Vec<NodeCase> {
    let mut operations = BTreeSet::new();
    cases
        .iter()
        .filter(|case| {
            case.value_input.is_some()
                || (!is_eevee_probe(case)
                    && (case.clamp || operations.insert((case.kind, case.operation))))
        })
        .cloned()
        .collect()
}

fn eevee_probe_cases(cases: &[NodeCase]) -> Vec<NodeCase> {
    cases
        .iter()
        .filter(|case| is_eevee_probe(case))
        .cloned()
        .collect()
}

fn blender_reference(cases: &[NodeCase], include_shader: bool) -> BlenderOracle {
    let blender = blender_file::blender_executable().expect("Blender is available for this test");
    let directory = tempdir().unwrap();
    let cases_path = directory.path().join("cases.json");
    let shader_blend_path = directory.path().join("shader_node_math.blend");
    let eevee_blend_path = directory.path().join("shader_node_math_eevee.blend");
    let script_path =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/node_math/generate_oracle.py");
    let shader_blend_arg = if include_shader {
        shader_blend_path.as_path()
    } else {
        Path::new("-")
    };
    let eevee_blend_arg = if include_shader {
        eevee_blend_path.as_path()
    } else {
        Path::new("-")
    };
    let cli_cases = shader_cli_cases(cases);
    let cli_ids: BTreeSet<String> = cli_cases.iter().map(|case| case.id.clone()).collect();
    let eevee_cases = eevee_probe_cases(cases);
    let eevee_ids: BTreeSet<String> = eevee_cases.iter().map(|case| case.id.clone()).collect();
    let cases_json = Value::Array(
        cases
            .iter()
            .map(|case| {
                let mut input = case.blender_input();
                input["shader_cli"] = json!(cli_ids.contains(&case.id));
                input["eevee_cli"] = json!(eevee_ids.contains(&case.id));
                input
            })
            .collect(),
    );
    std::fs::write(&cases_path, serde_json::to_vec(&cases_json).unwrap()).unwrap();
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
        .arg(shader_blend_arg)
        .arg(eevee_blend_arg)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "Blender math oracle failed:\n{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    let json_line = stdout
        .lines()
        .find_map(|line| line.strip_prefix("POTTER_NODE_MATH="))
        .expect("Blender oracle emitted a result line");
    let reference: Value = serde_json::from_str(json_line).unwrap();
    BlenderOracle {
        directory,
        geometry: reference["geometry"].as_object().unwrap().clone(),
        shader: reference["shader"].as_object().unwrap().clone(),
        eevee: reference["eevee"].as_object().unwrap().clone(),
        shader_blend_path,
        eevee_blend_path,
    }
}

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
        (Value::Array(actual), Value::Array(expected)) if actual.len() == expected.len() => actual
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

struct ProbeImage {
    width: usize,
    height: usize,
    pixels: Vec<[f64; 3]>,
}

fn read_probe_image(path: &Path) -> ProbeImage {
    let image = exr::prelude::read_first_rgba_layer_from_file(
        path,
        |resolution, _| ProbeImage {
            width: resolution.width(),
            height: resolution.height(),
            pixels: vec![[0.0; 3]; resolution.width() * resolution.height()],
        },
        |image: &mut ProbeImage, position, (red, green, blue, _): (f32, f32, f32, f32)| {
            image.pixels[position.y() * image.width + position.x()] =
                [f64::from(red), f64::from(green), f64::from(blue)];
        },
    )
    .unwrap();
    image.layer_data.channel_data.pixels
}
fn realtime_film_precision(value: f64) -> f64 {
    f64::from(exr::prelude::f16::from_f32(value as f32).to_f32())
}

fn pot_json(command: &mut Command, label: &str) -> Value {
    let output = command.output().unwrap();
    assert!(
        output.status.success(),
        "Potter {label} failed:\n{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );
    serde_json::from_slice(&output.stdout).unwrap()
}

fn compare_shader_cli(oracle: &BlenderOracle, cases: &[NodeCase]) {
    let blender = blender_file::blender_executable().expect("Blender is available for this test");
    let probe_cases = shader_cli_cases(cases);
    let realtime_cases: Vec<NodeCase> = eevee_probe_cases(cases)
        .into_iter()
        .filter(|case| !oracle.eevee[&case.id].is_string())
        .collect();

    let project = oracle.directory.path().join("shader_cli_project");
    let project_text = project.to_str().unwrap();
    let initialized = pot_json(
        Command::new(env!("CARGO_BIN_EXE_pot")).args(["init", project_text, "--json"]),
        "shader project initialization",
    );
    assert_eq!(initialized["ok"], true);
    let imported = pot_json(
        Command::new(env!("CARGO_BIN_EXE_pot"))
            .args(["import", project_text, "--file"])
            .arg(&oracle.shader_blend_path)
            .args([
                "--format",
                "blend",
                "--mode",
                "replace",
                "--base-revision",
                "0",
                "--blender",
            ])
            .arg(&blender)
            .arg("--json"),
        "Cycles shader probe import",
    );
    assert_eq!(imported["ok"], true);

    let output_directory = oracle.directory.path().join("shader_cli_render");
    let rendered = pot_json(
        Command::new(env!("CARGO_BIN_EXE_pot"))
            .args([
                "render",
                project_text,
                "--engine",
                "path",
                "--format",
                "exr",
                "--out",
            ])
            .arg(&output_directory)
            .arg("--json"),
        "shader surface CLI render",
    );
    let frame_path = rendered["result"]["frames"][0]["path"]
        .as_str()
        .expect("CLI render returned a frame path");
    let image = read_probe_image(Path::new(frame_path));
    assert_eq!(image.width, probe_cases.len() * 8);
    assert_eq!(image.height, 8);

    let mut mismatches = Vec::new();
    for (column, case) in probe_cases.iter().enumerate() {
        let pixel = image.pixels[4 * image.width + column * 8 + 4];
        let actual = if case.data_type == "FLOAT_VECTOR" {
            json!(pixel)
        } else {
            json!(pixel[0])
        };
        let expected = &oracle.shader[&case.id];
        if expected.is_string() {
            mismatches.push(format!(
                "shader_cli_{}: Cycles output is non-finite",
                case.id
            ));
        } else if let Some(mismatch) =
            json_mismatch(&actual, expected, &format!("shader_cli_{}", case.id))
        {
            mismatches.push(mismatch);
        }
    }
    assert!(
        mismatches.is_empty(),
        "{} of {} exact-pixel CLI shader probes diverge from Blender Cycles:\n{}",
        mismatches.len(),
        probe_cases.len(),
        mismatches.join("\n")
    );

    let realtime_project = oracle.directory.path().join("shader_realtime_project");
    let realtime_project_text = realtime_project.to_str().unwrap();
    let initialized = pot_json(
        Command::new(env!("CARGO_BIN_EXE_pot")).args(["init", realtime_project_text, "--json"]),
        "realtime shader project initialization",
    );
    assert_eq!(initialized["ok"], true);
    let imported = pot_json(
        Command::new(env!("CARGO_BIN_EXE_pot"))
            .args(["import", realtime_project_text, "--file"])
            .arg(&oracle.eevee_blend_path)
            .args([
                "--format",
                "blend",
                "--mode",
                "replace",
                "--base-revision",
                "0",
                "--blender",
            ])
            .arg(&blender)
            .arg("--json"),
        "realtime shader probe import",
    );
    assert_eq!(imported["ok"], true);

    let realtime_directory = oracle.directory.path().join("shader_realtime_render");
    let rendered = pot_json(
        Command::new(env!("CARGO_BIN_EXE_pot"))
            .args([
                "render",
                realtime_project_text,
                "--engine",
                "realtime",
                "--format",
                "exr",
                "--out",
            ])
            .arg(&realtime_directory)
            .arg("--json"),
        "realtime shader CLI render",
    );
    let frame_path = rendered["result"]["frames"][0]["path"]
        .as_str()
        .expect("CLI realtime render returned a frame path");
    let image = read_probe_image(Path::new(frame_path));
    assert_eq!(image.width, realtime_cases.len() * 8);
    assert_eq!(image.height, 8);

    let mut mismatches = Vec::new();
    for (column, case) in realtime_cases.iter().enumerate() {
        let pixel = image.pixels[4 * image.width + column * 8 + 4];
        let actual = if case.data_type == "FLOAT_VECTOR" {
            json!(pixel.map(realtime_film_precision))
        } else {
            json!(realtime_film_precision(pixel[0]))
        };
        let expected = &oracle.eevee[&case.id];
        let mismatch = if case.id == "eevee_math_arcsine_in_domain" {
            let actual = actual.as_f64().unwrap();
            let expected = expected.as_f64().unwrap();
            let tolerance = 5.0e-6;
            ((actual - expected).abs() > tolerance).then(|| {
                format!(
                    "realtime_cli_{}: Blender {expected:?}, Potter {actual:?} (tolerance {tolerance})",
                    case.id
                )
            })
        } else {
            json_mismatch(&actual, expected, &format!("realtime_cli_{}", case.id))
        };
        if let Some(mismatch) = mismatch {
            mismatches.push(mismatch);
        }
    }
    assert!(
        mismatches.is_empty(),
        "{} of {} exact-pixel realtime CLI shader probes diverge from Blender EEVEE:\n{}",
        mismatches.len(),
        realtime_cases.len(),
        mismatches.join("\n")
    );
}

#[test]
fn realtime_shader_math_dynamic_inputs_match_blender_522() {
    let Some(blender) = blender_file::blender_executable() else {
        eprintln!("Skipping dynamic realtime shader parity: no Blender executable was found");
        return;
    };
    let directory = tempdir().unwrap();
    let blend_path = directory.path().join("dynamic_shader_math.blend");
    let script = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/node_math/generate_realtime_dynamic_probes.py");
    let generated = Command::new(&blender)
        .args([
            "--background",
            "--factory-startup",
            "--python-exit-code",
            "1",
            "--python",
        ])
        .arg(script)
        .arg("--")
        .arg(&blend_path)
        .output()
        .unwrap();
    assert!(
        generated.status.success(),
        "Blender dynamic realtime shader oracle failed:\n{}\n{}",
        String::from_utf8_lossy(&generated.stdout),
        String::from_utf8_lossy(&generated.stderr),
    );
    let stdout = String::from_utf8_lossy(&generated.stdout);
    let oracle_line = stdout
        .lines()
        .find_map(|line| line.strip_prefix("POTTER_NODE_MATH_REALTIME_DYNAMIC="))
        .expect("Blender dynamic realtime oracle emitted a result line");
    let oracle: Value = serde_json::from_str(oracle_line).unwrap();
    let captured: Value = serde_json::from_str(include_str!(
        "fixtures/node_math/realtime_dynamic_eevee.json"
    ))
    .unwrap();
    assert_eq!(oracle["engine"], captured["engine"]);
    assert_eq!(oracle["resolution"], captured["resolution"]);
    let cases = oracle["cases"].as_array().unwrap();
    let captured_cases = captured["cases"].as_array().unwrap();
    assert_eq!(cases.len(), captured_cases.len());
    for (case, expected_case) in cases.iter().zip(captured_cases) {
        assert_eq!(case["id"], expected_case["id"]);
        assert_eq!(case["kind"], expected_case["kind"]);
    }
    assert_eq!(
        oracle["eevee"]
            .as_object()
            .unwrap()
            .keys()
            .collect::<Vec<_>>(),
        captured["eevee"]
            .as_object()
            .unwrap()
            .keys()
            .collect::<Vec<_>>(),
    );
    let project = directory.path().join("dynamic_shader_project");
    let project_text = project.to_str().unwrap();
    let initialized = pot_json(
        Command::new(env!("CARGO_BIN_EXE_pot")).args(["init", project_text, "--json"]),
        "dynamic realtime shader project initialization",
    );
    assert_eq!(initialized["ok"], true);
    let imported = pot_json(
        Command::new(env!("CARGO_BIN_EXE_pot"))
            .args(["import", project_text, "--file"])
            .arg(&blend_path)
            .args([
                "--format",
                "blend",
                "--mode",
                "replace",
                "--base-revision",
                "0",
                "--blender",
            ])
            .arg(&blender)
            .arg("--json"),
        "dynamic realtime shader import",
    );
    assert_eq!(imported["ok"], true);

    let output_directory = directory.path().join("dynamic_shader_render");
    let rendered = pot_json(
        Command::new(env!("CARGO_BIN_EXE_pot"))
            .args([
                "render",
                project_text,
                "--engine",
                "realtime",
                "--format",
                "exr",
                "--out",
            ])
            .arg(&output_directory)
            .arg("--json"),
        "dynamic realtime shader render",
    );
    let frame_path = rendered["result"]["frames"][0]["path"]
        .as_str()
        .expect("CLI dynamic realtime render returned a frame path");
    let image = read_probe_image(Path::new(frame_path));
    assert_eq!(image.width, cases.len() * 8);
    assert_eq!(image.height, 8);

    let mut mismatches = Vec::new();
    for (column, case) in cases.iter().enumerate() {
        let pixel = image.pixels[4 * image.width + column * 8 + 4];
        let actual = if case["kind"] == "scalar" {
            json!(realtime_film_precision(pixel[0]))
        } else {
            json!(pixel.map(realtime_film_precision))
        };
        let id = case["id"].as_str().unwrap();
        let expected = &captured["eevee"][id];
        match (&actual, expected) {
            (Value::Number(actual), Value::Number(expected)) => assert_eevee_film_close(
                actual.as_f64().unwrap(),
                expected.as_f64().unwrap(),
                &format!("dynamic_realtime_{id}"),
            ),
            (Value::Array(actual), Value::Array(expected)) if actual.len() == expected.len() => {
                for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
                    assert_eevee_film_close(
                        actual.as_f64().unwrap(),
                        expected.as_f64().unwrap(),
                        &format!("dynamic_realtime_{id}[{index}]"),
                    );
                }
            }
            _ => mismatches.push(format!(
                "dynamic_realtime_{id}: committed Blender and Potter returned different value shapes"
            )),
        }
    }
    assert!(
        mismatches.is_empty(),
        "{} of {} per-pixel realtime shader probes diverge from the committed Blender 5.2.2 EEVEE capture:\n{}",
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
    let oracle = blender_reference(&cases, false);
    compare_cases(
        &cases,
        &oracle.geometry,
        "geometry_",
        evaluate_geometry_case,
        |_| true,
    );
}

#[test]
fn shader_math_and_vector_math_match_cycles_and_cli() {
    let Some(_) = blender_file::blender_executable() else {
        eprintln!("Skipping Blender shader math parity: no Blender executable was found");
        return;
    };
    let cases: Vec<NodeCase> = math_cases().into_iter().chain(vector_cases()).collect();
    let oracle = blender_reference(&cases, true);
    compare_cases(
        &cases,
        &oracle.shader,
        "shader_",
        evaluate_shader_case,
        |case| {
            case.value_input.is_some()
                || (!is_eevee_probe(case)
                    && !matches!(
                        case.id.as_str(),
                        "math_exponent_overflow"
                            | "vector_power_overflow"
                            | "math_power_zero_negative"
                    ))
        },
    );
    let nonfinite_power = cases
        .iter()
        .find(|case| case.id == "math_power_zero_negative")
        .expect("zero-to-negative shader power probe exists");
    let error = evaluate_shader_case(nonfinite_power).unwrap_err();
    assert_eq!(error.code, ErrorCode::UnsupportedFeature);
    assert_eq!(
        error.details["feature_id"],
        "shader.node.ShaderNodeMath.operation.POWER.non_finite_constant_fold"
    );
    for case in cases.iter().filter(|case| {
        matches!(
            case.id.as_str(),
            "math_exponent_overflow" | "vector_power_overflow"
        )
    }) {
        assert!(
            evaluate_shader_case(case).is_err(),
            "overflowing shader result should be refused: {}",
            case.id
        );
    }
    compare_shader_cli(&oracle, &cases);
}

fn blender_node_state(blender: &Path, script: &Path, mode: &str, blend_path: &Path) -> Value {
    let output = Command::new(blender)
        .args([
            "--background",
            "--factory-startup",
            "--python-exit-code",
            "1",
            "--python",
        ])
        .arg(script)
        .arg("--")
        .arg(mode)
        .arg(blend_path)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "Blender node-state {mode} failed:\n{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    let line = stdout
        .lines()
        .find_map(|line| line.strip_prefix("POTTER_NODE_STATE="))
        .expect("Blender state script emitted its JSON result");
    serde_json::from_str(line).unwrap()
}

#[test]
fn shader_curve_mappings_and_color_ramp_survive_blend_cli_round_trip() {
    let Some(blender) = blender_file::blender_executable() else {
        eprintln!("Skipping node-state round trip: no Blender executable was found");
        return;
    };
    let directory = tempdir().unwrap();
    let root = directory.path();
    let script = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/node_math/generate_state_roundtrip.py");
    let source_blend = root.join("node_state.blend");
    let source = blender_node_state(&blender, &script, "generate", &source_blend);
    assert_eq!(source["float_curve"]["curves"].as_array().unwrap().len(), 1);
    assert_eq!(
        source["vector_curve"]["curves"].as_array().unwrap().len(),
        3
    );
    assert_eq!(source["rgb_curve"]["curves"].as_array().unwrap().len(), 4);
    assert_eq!(source["float_curve"]["extend"], "HORIZONTAL");
    assert_eq!(source["float_curve"]["use_clip"], false);
    assert_eq!(source["vector_curve"]["extend"], "HORIZONTAL");
    assert_eq!(source["vector_curve"]["use_clip"], false);
    assert_eq!(source["rgb_curve"]["tone"], "FILMLIKE");
    assert_eq!(source["rgb_curve"]["extend"], "HORIZONTAL");
    assert_eq!(source["rgb_curve"]["use_clip"], true);
    assert_eq!(
        source["rgb_curve"]["clip"],
        json!([0.125, 0.0625, 0.875, 0.9375])
    );
    assert_eq!(source["rgb_curve"]["black"], json!([0.0625, 0.125, 0.1875]));
    assert_eq!(source["rgb_curve"]["white"], json!([0.8125, 0.875, 0.9375]));
    assert_eq!(
        source["rgb_curve"]["curves"][3][1]["handle"],
        "AUTO_CLAMPED"
    );
    assert_eq!(source["ramp"]["color_mode"], "HSL");
    assert_eq!(source["ramp"]["hue_interpolation"], "CW");
    assert_eq!(source["ramp"]["interpolation"], "EASE");
    assert_eq!(source["ramp"]["elements"].as_array().unwrap().len(), 3);

    let project = root.join("project");
    let project_text = project.to_str().unwrap();
    let initialized = pot_json(
        Command::new(env!("CARGO_BIN_EXE_pot")).args(["init", project_text, "--json"]),
        "node-state project initialization",
    );
    assert_eq!(initialized["ok"], true);
    let imported = pot_json(
        Command::new(env!("CARGO_BIN_EXE_pot"))
            .args(["import", project_text, "--file"])
            .arg(&source_blend)
            .args([
                "--format",
                "blend",
                "--mode",
                "replace",
                "--base-revision",
                "0",
                "--blender",
            ])
            .arg(&blender)
            .arg("--json"),
        "node-state Blender import",
    );
    assert_eq!(imported["ok"], true);

    let exported_blend = root.join("node_state_round_trip.blend");
    let exported = pot_json(
        Command::new(env!("CARGO_BIN_EXE_pot"))
            .args(["export", project_text, "--format", "blend", "--out"])
            .arg(&exported_blend)
            .args(["--blender"])
            .arg(&blender)
            .arg("--json"),
        "node-state Blender export",
    );
    assert_eq!(exported["ok"], true);
    let round_trip = blender_node_state(&blender, &script, "read", &exported_blend);
    assert_eq!(
        round_trip, source,
        "shader node storage changed across .blend round trip"
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
    let shading = evaluate_surface(&material, &HitContext::new(&groups))?;
    Ok(*shading
        .dominant()
        .expect("a Principled surface has one closure"))
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
fn shader_math_supports_sign_and_reflect_and_rejects_wrong_sockets() {
    let sign = shader_surface_with_node(
        "ShaderNodeMath",
        json!({"Value": -2.5}),
        json!({"operation": "SIGN"}),
        "Value",
        "Emission Strength",
    )
    .unwrap();
    assert_eq!(sign.emission_strength, -1.0);

    let reflect = shader_surface_with_node(
        "ShaderNodeVectorMath",
        json!({
            "Vector": [1.0, 2.0, 3.0],
            "Vector_001": [0.0, 1.0, 0.0],
        }),
        json!({"operation": "REFLECT"}),
        "Vector",
        "Emission Color",
    )
    .unwrap();
    assert_eq!(reflect.emission_color, [1.0, -2.0, 3.0]);

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
        json!({"Value": 1.0, "Value_001": 1.25}),
        json!({"operation": "COMPARE"}),
        "Value",
        "Emission Strength",
    )
    .unwrap();
    assert!((within.emission_strength - 1.0).abs() < f64::EPSILON);
    let outside = shader_surface_with_node(
        "ShaderNodeMath",
        json!({"Value": 1.0, "Value_001": 1.75}),
        json!({"operation": "COMPARE"}),
        "Value",
        "Emission Strength",
    )
    .unwrap();
    assert!(outside.emission_strength.abs() < f64::EPSILON);
}
#[test]
fn shader_map_range_defaults_to_blenders_enabled_clamp() {
    let params = shader_surface_with_node(
        "ShaderNodeMapRange",
        json!({
            "Value": 2.0,
            "From Min": 0.0,
            "From Max": 1.0,
            "To Min": 0.25,
            "To Max": 0.75
        }),
        json!({}),
        "Result",
        "Emission Strength",
    )
    .unwrap();
    assert_eq!(params.emission_strength, 0.75);
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
#[test]
fn realtime_shader_math_guards_report_exact_feature_ids() {
    let Some(blender) = blender_file::blender_executable() else {
        eprintln!("Skipping realtime shader guard parity: no Blender executable was found");
        return;
    };
    let directory = tempdir().unwrap();
    let script = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/node_math/generate_realtime_refusals.py");
    let fixtures = directory.path().join("fixtures");
    let generated = Command::new(&blender)
        .args([
            "--background",
            "--factory-startup",
            "--python-exit-code",
            "1",
            "--python",
        ])
        .arg(script)
        .arg("--")
        .arg(&fixtures)
        .output()
        .unwrap();
    assert!(
        generated.status.success(),
        "Blender realtime shader guards failed:\n{}\n{}",
        String::from_utf8_lossy(&generated.stdout),
        String::from_utf8_lossy(&generated.stderr),
    );

    for (index, (fixture_id, feature_id)) in [
        (
            "shader_math_modulo_quotient_out_of_range",
            "shader.node.ShaderNodeMath.operation.MODULO.quotient_out_of_range",
        ),
        (
            "shader_vector_math_modulo_quotient_out_of_range",
            "shader.node.ShaderNodeVectorMath.operation.MODULO.quotient_out_of_range",
        ),
        (
            "shader_vector_math_normalize_subnormal_input",
            "shader.node.ShaderNodeVectorMath.operation.NORMALIZE.subnormal_input",
        ),
        (
            "shader_math_inverse_sqrt_nonpositive",
            "shader.node.ShaderNodeMath.operation.INVERSE_SQRT.glsl_undefined_domain",
        ),
        (
            "shader_math_logarithm_base_one",
            "shader.node.ShaderNodeMath.operation.LOGARITHM.glsl_undefined_domain",
        ),
    ]
    .into_iter()
    .enumerate()
    {
        let project = directory.path().join(format!("project_{index}"));
        let project_text = project.to_str().unwrap();
        let initialized = pot_json(
            Command::new(env!("CARGO_BIN_EXE_pot")).args(["init", project_text, "--json"]),
            "realtime guard project initialization",
        );
        assert_eq!(initialized["ok"], true);
        let imported = pot_json(
            Command::new(env!("CARGO_BIN_EXE_pot"))
                .args(["import", project_text, "--file"])
                .arg(fixtures.join(format!("{fixture_id}.blend")))
                .args([
                    "--format",
                    "blend",
                    "--mode",
                    "replace",
                    "--base-revision",
                    "0",
                    "--blender",
                ])
                .arg(&blender)
                .arg("--json"),
            "realtime guard shader import",
        );
        assert_eq!(imported["ok"], true);

        let output_directory = directory.path().join(format!("render_{index}"));
        let output = Command::new(env!("CARGO_BIN_EXE_pot"))
            .args([
                "render",
                project_text,
                "--engine",
                "realtime",
                "--format",
                "exr",
                "--out",
            ])
            .arg(output_directory)
            .arg("--json")
            .output()
            .unwrap();
        assert!(
            !output.status.success(),
            "{fixture_id}: unsupported realtime shader math unexpectedly rendered"
        );
        let response: Value = serde_json::from_slice(&output.stdout).unwrap();
        assert_eq!(
            response["error"]["code"], "UNSUPPORTED_FEATURE",
            "{fixture_id}"
        );
        assert_eq!(
            response["error"]["details"]["feature_id"], feature_id,
            "{fixture_id}"
        );
    }
}
