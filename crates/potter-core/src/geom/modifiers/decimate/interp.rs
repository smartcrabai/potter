//! `CD_PROP_FLOAT2` loop interpolation used when collapsing edges: ports of Blender's
//! `layerInterp_propfloat2` and `layerEqual_propfloat2` (customdata.cc).

/// Weighted sum of `CD_PROP_FLOAT2` values.
///
/// Mirrors `layerInterp_propfloat2` (customdata.cc), which accumulates
/// `madd_v2_v2fl` from a zero vector in source order. Weights are used as given;
/// they are not normalized, matching `CustomData_bmesh_interp_n`.
pub(super) fn interp_float2(sources: &[[f32; 2]], weights: &[f32]) -> [f32; 2] {
    let mut result = [0.0_f32, 0.0_f32];
    for (src, &interp_weight) in sources.iter().zip(&weights[..sources.len()]) {
        result[0] += src[0] * interp_weight;
        result[1] += src[1] * interp_weight;
    }
    result
}

/// Equality test for `CD_PROP_FLOAT2` values.
///
/// Mirrors `layerEqual_propfloat2` (customdata.cc): squared distance
/// (`math::distance_squared`, i.e. `dot(a - b, a - b)`) strictly below `0.00001f`.
pub(super) fn float2_equal(a: [f32; 2], b: [f32; 2]) -> bool {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    dx * dx + dy * dy < 0.00001_f32
}
