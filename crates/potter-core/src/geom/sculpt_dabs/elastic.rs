//! Blender 5.2.2 Elastic Grab (`mesh/brushes/elastic_deform.cc`) on the dab framework.
//!
//! The essentials "Elastic Grab" asset uses `elastic_deform_type` = `GRAB_TRISCALE`, so Potter
//! ports that solver only: each vertex moves by the grab delta times the three-scale regularized
//! Kelvinlet fade (`BKE_kelvinlet_grab_triscale`, kelvinlet.cc). `dir` and `force` in
//! `do_elastic_deform_brush` feed only the SCALE and TWIST solvers, so they are not ported. The
//! brush has no falloff of its own: the Kelvinlet fade is its only spatial factor.

use std::f64::consts::PI;

use super::{DabInput, Field};
use crate::geom::sculpt::DabBrush;

/// `elastic_deform_volume_preservation` of the 5.2.2 essentials "Elastic Grab" asset (the DNA
/// default is 0). It is the Kelvinlet's Poisson ratio, passed to `BKE_kelvinlet_init_params`
/// (kelvinlet.cc).
const VOLUME_PRESERVATION: f64 = 0.4;
/// The `shear_modulus` argument of `BKE_kelvinlet_init_params` in `do_elastic_deform_brush`.
const SHEAR_MODULUS: f64 = 1.0;
/// `calc_translations` scales the Kelvinlet translations by `cache.bstrength * 20.0f`
/// (`elastic_deform.cc`).
const STRENGTH_SCALE: f64 = 20.0;

/// Blender's `BKE_kelvinlet_grab_triscale` (kelvinlet.cc): the scalar fade that multiplies the grab
/// delta for a vertex `distance` from the brush location. Three Kelvinlets at the scales `radius`,
/// `2 radius` and `4 radius` (`BKE_kelvinlet_init_params` `radius_e` = {1, 2, 2}) are combined with
/// the weights that cancel the first two scale terms.
fn triscale_fade(distance: f64, radius: f64) -> f64 {
    // `KelvinletParams` a, b, c (`BKE_kelvinlet_init_params`).
    let coef_a = 1.0 / (4.0 * PI * SHEAR_MODULUS);
    let coef_b = coef_a / (4.0 * (1.0 - VOLUME_PRESERVATION));
    let coef_c = 2.0 * (3.0 * coef_a - 2.0 * coef_b);
    let scales = [radius, 2.0 * radius, 4.0 * radius];
    // Regularized Kelvinlets: Formula (6), `init_kelvinlet_grab`.
    let kelvinlets = scales.map(|scale| {
        let root = (distance * distance + scale * scale).sqrt();
        let cube = root.powi(3);
        (coef_a - coef_b) / root
            + coef_b * distance * distance / cube
            + coef_a * scale * scale / (2.0 * cube)
    });
    // The `BKE_kelvinlet_grab_triscale` weights, with the scales squared.
    let squares = scales.map(|scale| scale * scale);
    let span = squares[2] - squares[1];
    let weights = [
        1.0,
        -(squares[2] - squares[0]) / span,
        (squares[1] - squares[0]) / span,
    ];
    let numerator: f64 = weights
        .iter()
        .zip(kelvinlets)
        .map(|(weight, value)| weight * value)
        .sum();
    let inverse_scale: f64 = weights
        .iter()
        .zip(scales)
        .map(|(weight, scale)| weight / scale)
        .sum();
    numerator * coef_c / inverse_scale
}

/// Blender 5.2.2 Elastic Grab for one dab (`do_elastic_deform_brush`, `GRAB_TRISCALE`), run once
/// per symmetry pass.
///
/// `input.center` is the stroke-start location and `input.grab` the grab from it, as the framework
/// supplies for anchored brushes (`brush_delta_update`, sculpt.cc:4310-4312 and 4278-4281). The
/// framework resets the positions to the stroke-start state before this dab
/// (`restore_from_undo_step_if_necessary`, sculpt.cc:5160-5180), so a dab replaces the displacement
/// of the previous one. Each pass then moves every vertex by its grab times `triscale_fade`, the
/// `bstrength * 20` scale, and `1 - mask`. The fade reads the stroke-start positions
/// (`orig_position_data_get_mesh`), and the mirror pass flips the location and the grab
/// (`location_symm`, `grab_delta_symm`).
pub(super) fn apply(field: &mut Field, dab: DabBrush, input: &DabInput) {
    let DabBrush::Elastic = dab else {
        unreachable!("elastic::apply only handles the Elastic brush");
    };
    let scale = input.strength * STRENGTH_SCALE;
    for sign in &field.passes {
        let location = input.center * *sign;
        let grab = input.grab * *sign;
        for (index, original) in field.original.iter().enumerate() {
            let fade = triscale_fade(original.distance(location), input.radius);
            field.positions[index] += grab * (fade * scale * field.mask_factors[index]);
        }
    }
}
