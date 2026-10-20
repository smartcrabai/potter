#![expect(
    clippy::float_cmp,
    reason = "asserting exact identity of the Raw display transform inside the display range"
)]

use potter_core::color::curve::{CurveMapping, CurvePoint, HandleType};
use potter_core::color::{ColorManagement, ViewTransform, linear_to_srgb};
use proptest::prelude::*;

/// Runs `display_encode` for settings that are supported (sRGB display, look `none`).
fn encode(settings: &ColorManagement, rgb: [f32; 3]) -> [f32; 3] {
    settings
        .display_encode(rgb)
        .unwrap_or_else(|error| panic!("display_encode failed: {error}"))
}

proptest! {
    #[test]
    fn raw_is_identity_inside_the_display_range_and_clamps_outside(
        rgb in prop::array::uniform3(-1000.0_f32..1000.0)
    ) {
        let settings = ColorManagement {
            view_transform: ViewTransform::Raw,
            ..ColorManagement::default()
        };
        prop_assert_eq!(encode(&settings, rgb), rgb.map(|value| value.clamp(0.0, 1.0)));
    }

    #[test]
    fn agx_filmic_gamma_and_srgb_are_monotone(
        value in 0.0_f32..100.0,
        delta in 0.0_f32..10.0
    ) {
        for view_transform in [ViewTransform::AgX, ViewTransform::Filmic] {
            let settings = ColorManagement {
                view_transform,
                ..ColorManagement::default()
            };
            let lower = encode(&settings, [value; 3]);
            let upper = encode(&settings, [value + delta; 3]);
            for (low, high) in lower.into_iter().zip(upper) {
                prop_assert!(high + 1.0e-5 >= low, "{view_transform:?}: {high} < {low}");
            }
        }

        let gamma = ColorManagement {
            view_transform: ViewTransform::Raw,
            gamma: 2.5,
            ..ColorManagement::default()
        };
        let lower = encode(&gamma, [value; 3]);
        let upper = encode(&gamma, [value + delta; 3]);
        for (low, high) in lower.into_iter().zip(upper) {
            prop_assert!(high + 1.0e-5 >= low);
        }
        let (lo, hi) = (f64::from(value), f64::from(value + delta));
        prop_assert!(linear_to_srgb(hi) + 1.0e-12 >= linear_to_srgb(lo));
    }

    #[test]
    fn monotone_curve_preserves_channel_order(
        lower in 0.0_f32..0.9,
        delta in 0.0_f32..0.1
    ) {
        let upper = (lower + delta).min(1.0);
        let mut combined = CurveMapping::default();
        combined.curves[3].points = [
            ([0.0, 0.0], HandleType::Auto),
            ([0.5, 0.25], HandleType::Auto),
            ([1.0, 1.0], HandleType::Auto),
        ]
        .into_iter()
        .map(|(location, handle)| CurvePoint { location, handle })
        .collect();
        let settings = ColorManagement {
            view_transform: ViewTransform::Raw,
            curve: Some(combined),
            ..ColorManagement::default()
        };
        let lower = encode(&settings, [lower; 3]);
        let upper = encode(&settings, [upper; 3]);
        for (low, high) in lower.into_iter().zip(upper) {
            prop_assert!(high + 1.0e-6 >= low);
        }
    }
}
