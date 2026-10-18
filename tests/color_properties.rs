#![expect(
    clippy::float_cmp,
    reason = "asserting exact identity of display transforms"
)]

use potter_core::color::{ColorManagement, ViewTransform, linear_to_srgb};
use proptest::prelude::*;

/// Runs `transform_linear` for settings that are supported (sRGB display, look `none`).
fn transform(settings: &ColorManagement, rgb: [f64; 3]) -> [f64; 3] {
    settings
        .transform_linear(rgb)
        .unwrap_or_else(|error| panic!("transform_linear failed: {error}"))
}

proptest! {
    #[test]
    fn raw_and_standard_transforms_are_identity_for_finite_channels(
        rgb in prop::array::uniform3(-1000.0_f64..1000.0)
    ) {
        for view_transform in [ViewTransform::Raw, ViewTransform::Standard] {
            let settings = ColorManagement {
                view_transform,
                ..ColorManagement::default()
            };
            prop_assert_eq!(transform(&settings, rgb), rgb);
        }
    }

    #[test]
    fn agx_filmic_gamma_and_srgb_are_monotone(
        value in 0.0_f64..100.0,
        delta in 0.0_f64..10.0
    ) {
        for view_transform in [ViewTransform::AgX, ViewTransform::Filmic] {
            let settings = ColorManagement {
                view_transform,
                ..ColorManagement::default()
            };
            let lower = transform(&settings, [value; 3]);
            let upper = transform(&settings, [value + delta; 3]);
            for (low, high) in lower.into_iter().zip(upper) {
                prop_assert!(high + 1.0e-12 >= low);
            }
        }

        let gamma = ColorManagement {
            view_transform: ViewTransform::Raw,
            gamma: 2.5,
            ..ColorManagement::default()
        };
        let lower = transform(&gamma, [value; 3]);
        let upper = transform(&gamma, [value + delta; 3]);
        for (low, high) in lower.into_iter().zip(upper) {
            prop_assert!(high + 1.0e-12 >= low);
        }
        prop_assert!(linear_to_srgb(value + delta) + 1.0e-12 >= linear_to_srgb(value));
    }

    #[test]
    fn monotone_curve_preserves_channel_order(
        lower in 0.0_f64..0.9,
        delta in 0.0_f64..0.1
    ) {
        let upper = (lower + delta).min(1.0);
        let settings = ColorManagement {
            view_transform: ViewTransform::Raw,
            curve: vec![[0.0, 0.0], [0.5, 0.25], [1.0, 1.0]],
            ..ColorManagement::default()
        };
        let lower = transform(&settings, [lower; 3]);
        let upper = transform(&settings, [upper; 3]);
        for (low, high) in lower.into_iter().zip(upper) {
            prop_assert!(high + 1.0e-12 >= low);
        }
    }
}
