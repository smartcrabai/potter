//! Blender 5.2.2 colour-management looks for Potter's views, on display `sRGB`.
//!
//! A look is an OCIO `LookTransform` that Blender applies in the look's process space before the view.
//! Blender offers a look for a view only when the look's name matches the view's prefix:
//! - `Standard`, `Filmic` and `Raw` take the unprefixed `Filmic Log` looks (`Raw` is a data view, so the
//!   look still applies as a round trip);
//! - `AgX` takes the `AgX - *` looks, and `False Color` takes the `False Color - *` looks. Both use the
//!   `AgX Log` process space.
//!
//! The arithmetic of each look is in `ocio.rs`; this module names the looks and says what each one does.

use super::ViewTransform;

/// A look Potter supports, or no look.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Look {
    None,
    VeryHighContrast,
    HighContrast,
    MediumHighContrast,
    MediumContrast,
    MediumLowContrast,
    LowContrast,
    VeryLowContrast,
    AgxPunchy,
    AgxGreyscale,
    AgxVeryHighContrast,
    AgxHighContrast,
    AgxMediumHighContrast,
    AgxBaseContrast,
    AgxMediumLowContrast,
    AgxLowContrast,
    AgxVeryLowContrast,
    FalseColorPunchy,
    FalseColorGreyscale,
    FalseColorVeryHighContrast,
    FalseColorHighContrast,
    FalseColorMediumHighContrast,
    FalseColorBaseContrast,
    FalseColorMediumLowContrast,
    FalseColorLowContrast,
    FalseColorVeryLowContrast,
}

/// The `filmic_to_*` curve a `Filmic Log` look applies before the view's own curve.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum FilmicCurve {
    VeryHigh,
    High,
    MediumHigh,
    MediumLow,
    Low,
    VeryLow,
}

/// One `AgX Log` look operation (Blender's `AgX` and `False Color` look definitions).
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) enum AgxOp {
    /// `Punchy`: a shadow lift (`GradingTone`) and a power of 1.0912 (CDL, no clamp).
    Punchy,
    /// `Greyscale`: the luma of the linear values, in the log space.
    Greyscale,
    /// `Base Contrast`: an identity grade, which Blender's optimiser removes.
    Base,
    /// `GradingPrimary` in log style: contrast, pivot 0.4, and saturation when it is not 1.
    Grade {
        contrast: f32,
        saturation: Option<f32>,
    },
}

/// The operation a look applies in its process space.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) enum LookOps {
    /// No look.
    None,
    /// `Filmic Log`: a curve, or `None` for `Medium Contrast`, which Blender defines with no children.
    Filmic(Option<FilmicCurve>),
    /// `AgX Log`: one `AgxOp`.
    Agx(AgxOp),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Family {
    None,
    Filmic,
    Agx,
    FalseColor,
}

impl Look {
    /// Parses a look token (the registry names: `none`, `very_high_contrast`, `agx_punchy`,
    /// `false_color_greyscale`, and so on). Returns `None` for an unknown token.
    #[must_use]
    pub fn parse(token: &str) -> Option<Self> {
        Some(match token {
            "none" => Self::None,
            "very_high_contrast" => Self::VeryHighContrast,
            "high_contrast" => Self::HighContrast,
            "medium_high_contrast" => Self::MediumHighContrast,
            "medium_contrast" => Self::MediumContrast,
            "medium_low_contrast" => Self::MediumLowContrast,
            "low_contrast" => Self::LowContrast,
            "very_low_contrast" => Self::VeryLowContrast,
            "agx_punchy" => Self::AgxPunchy,
            "agx_greyscale" => Self::AgxGreyscale,
            "agx_very_high_contrast" => Self::AgxVeryHighContrast,
            "agx_high_contrast" => Self::AgxHighContrast,
            "agx_medium_high_contrast" => Self::AgxMediumHighContrast,
            "agx_base_contrast" => Self::AgxBaseContrast,
            "agx_medium_low_contrast" => Self::AgxMediumLowContrast,
            "agx_low_contrast" => Self::AgxLowContrast,
            "agx_very_low_contrast" => Self::AgxVeryLowContrast,
            "false_color_punchy" => Self::FalseColorPunchy,
            "false_color_greyscale" => Self::FalseColorGreyscale,
            "false_color_very_high_contrast" => Self::FalseColorVeryHighContrast,
            "false_color_high_contrast" => Self::FalseColorHighContrast,
            "false_color_medium_high_contrast" => Self::FalseColorMediumHighContrast,
            "false_color_base_contrast" => Self::FalseColorBaseContrast,
            "false_color_medium_low_contrast" => Self::FalseColorMediumLowContrast,
            "false_color_low_contrast" => Self::FalseColorLowContrast,
            "false_color_very_low_contrast" => Self::FalseColorVeryLowContrast,
            _ => return None,
        })
    }

    /// Whether Blender lists this look for `view`. `None` is listed for every view.
    #[must_use]
    pub fn is_compatible_with(self, view: ViewTransform) -> bool {
        match self.family() {
            Family::None => true,
            Family::Filmic => matches!(
                view,
                ViewTransform::Standard | ViewTransform::Filmic | ViewTransform::Raw
            ),
            Family::Agx => view == ViewTransform::AgX,
            Family::FalseColor => view == ViewTransform::FalseColor,
        }
    }

    /// The OCIO process space Blender converts into for this look, or `None` for no look. Tests check
    /// the registry against the config; the chain itself does not need the name.
    #[cfg(test)]
    pub fn process_space(self) -> Option<&'static str> {
        match self.family() {
            Family::None => None,
            Family::Filmic => Some("Filmic Log"),
            Family::Agx | Family::FalseColor => Some("AgX Log"),
        }
    }

    pub(super) fn ops(self) -> LookOps {
        match self {
            Self::None => LookOps::None,
            Self::VeryHighContrast => LookOps::Filmic(Some(FilmicCurve::VeryHigh)),
            Self::HighContrast => LookOps::Filmic(Some(FilmicCurve::High)),
            Self::MediumHighContrast => LookOps::Filmic(Some(FilmicCurve::MediumHigh)),
            Self::MediumContrast => LookOps::Filmic(None),
            Self::MediumLowContrast => LookOps::Filmic(Some(FilmicCurve::MediumLow)),
            Self::LowContrast => LookOps::Filmic(Some(FilmicCurve::Low)),
            Self::VeryLowContrast => LookOps::Filmic(Some(FilmicCurve::VeryLow)),
            Self::AgxPunchy | Self::FalseColorPunchy => LookOps::Agx(AgxOp::Punchy),
            Self::AgxGreyscale | Self::FalseColorGreyscale => LookOps::Agx(AgxOp::Greyscale),
            Self::AgxBaseContrast | Self::FalseColorBaseContrast => LookOps::Agx(AgxOp::Base),
            Self::AgxVeryHighContrast | Self::FalseColorVeryHighContrast => {
                LookOps::Agx(grade(1.57, Some(0.9)))
            }
            Self::AgxHighContrast | Self::FalseColorHighContrast => {
                LookOps::Agx(grade(1.4, Some(0.95)))
            }
            Self::AgxMediumHighContrast | Self::FalseColorMediumHighContrast => {
                LookOps::Agx(grade(1.2, None))
            }
            Self::AgxMediumLowContrast | Self::FalseColorMediumLowContrast => {
                LookOps::Agx(grade(0.9, Some(1.05)))
            }
            Self::AgxLowContrast | Self::FalseColorLowContrast => {
                LookOps::Agx(grade(0.8, Some(1.1)))
            }
            Self::AgxVeryLowContrast | Self::FalseColorVeryLowContrast => {
                LookOps::Agx(grade(0.7, Some(1.15)))
            }
        }
    }

    fn family(self) -> Family {
        match self {
            Self::None => Family::None,
            Self::VeryHighContrast
            | Self::HighContrast
            | Self::MediumHighContrast
            | Self::MediumContrast
            | Self::MediumLowContrast
            | Self::LowContrast
            | Self::VeryLowContrast => Family::Filmic,
            Self::AgxPunchy
            | Self::AgxGreyscale
            | Self::AgxVeryHighContrast
            | Self::AgxHighContrast
            | Self::AgxMediumHighContrast
            | Self::AgxBaseContrast
            | Self::AgxMediumLowContrast
            | Self::AgxLowContrast
            | Self::AgxVeryLowContrast => Family::Agx,
            Self::FalseColorPunchy
            | Self::FalseColorGreyscale
            | Self::FalseColorVeryHighContrast
            | Self::FalseColorHighContrast
            | Self::FalseColorMediumHighContrast
            | Self::FalseColorBaseContrast
            | Self::FalseColorMediumLowContrast
            | Self::FalseColorLowContrast
            | Self::FalseColorVeryLowContrast => Family::FalseColor,
        }
    }
}

const fn grade(contrast: f32, saturation: Option<f32>) -> AgxOp {
    AgxOp::Grade {
        contrast,
        saturation,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ALL_TOKENS: [&str; 26] = [
        "none",
        "very_high_contrast",
        "high_contrast",
        "medium_high_contrast",
        "medium_contrast",
        "medium_low_contrast",
        "low_contrast",
        "very_low_contrast",
        "agx_punchy",
        "agx_greyscale",
        "agx_very_high_contrast",
        "agx_high_contrast",
        "agx_medium_high_contrast",
        "agx_base_contrast",
        "agx_medium_low_contrast",
        "agx_low_contrast",
        "agx_very_low_contrast",
        "false_color_punchy",
        "false_color_greyscale",
        "false_color_very_high_contrast",
        "false_color_high_contrast",
        "false_color_medium_high_contrast",
        "false_color_base_contrast",
        "false_color_medium_low_contrast",
        "false_color_low_contrast",
        "false_color_very_low_contrast",
    ];

    #[test]
    fn every_token_parses_to_a_distinct_look() {
        let looks: Vec<Look> = ALL_TOKENS
            .iter()
            .map(|token| Look::parse(token).expect("token parses"))
            .collect();
        for (index, look) in looks.iter().enumerate() {
            assert!(!looks[..index].contains(look), "duplicate look {look:?}");
        }
        assert_eq!(Look::parse("None"), None);
        assert_eq!(Look::parse("agx_missing"), None);
    }

    #[test]
    fn filmic_looks_apply_to_filmic_standard_and_raw_only() {
        let filmic = [
            Look::VeryHighContrast,
            Look::HighContrast,
            Look::MediumHighContrast,
            Look::MediumContrast,
            Look::MediumLowContrast,
            Look::LowContrast,
            Look::VeryLowContrast,
        ];
        for look in filmic {
            assert_eq!(look.process_space(), Some("Filmic Log"));
            for view in [
                ViewTransform::Standard,
                ViewTransform::Filmic,
                ViewTransform::Raw,
            ] {
                assert!(look.is_compatible_with(view), "{look:?} with {view:?}");
            }
            assert!(!look.is_compatible_with(ViewTransform::AgX));
            assert!(!look.is_compatible_with(ViewTransform::FalseColor));
        }
    }

    #[test]
    fn agx_and_false_color_looks_apply_to_their_own_view_only() {
        let agx = [Look::AgxPunchy, Look::AgxGreyscale, Look::AgxBaseContrast];
        for look in agx {
            assert_eq!(look.process_space(), Some("AgX Log"));
            assert!(look.is_compatible_with(ViewTransform::AgX));
            assert!(!look.is_compatible_with(ViewTransform::FalseColor));
            assert!(!look.is_compatible_with(ViewTransform::Filmic));
            assert!(!look.is_compatible_with(ViewTransform::Standard));
        }
        let false_color = [Look::FalseColorPunchy, Look::FalseColorVeryLowContrast];
        for look in false_color {
            assert_eq!(look.process_space(), Some("AgX Log"));
            assert!(look.is_compatible_with(ViewTransform::FalseColor));
            assert!(!look.is_compatible_with(ViewTransform::AgX));
            assert!(!look.is_compatible_with(ViewTransform::Raw));
        }
    }

    #[test]
    fn no_look_is_compatible_with_every_view_and_has_no_process_space() {
        for view in [
            ViewTransform::Standard,
            ViewTransform::AgX,
            ViewTransform::Filmic,
            ViewTransform::Raw,
            ViewTransform::FalseColor,
        ] {
            assert!(Look::None.is_compatible_with(view));
        }
        assert_eq!(Look::None.process_space(), None);
    }

    #[test]
    fn look_operations_follow_blender_definitions() {
        assert_eq!(Look::MediumContrast.ops(), LookOps::Filmic(None));
        assert_eq!(
            Look::VeryHighContrast.ops(),
            LookOps::Filmic(Some(FilmicCurve::VeryHigh))
        );
        assert_eq!(Look::AgxBaseContrast.ops(), LookOps::Agx(AgxOp::Base));
        assert_eq!(
            Look::FalseColorVeryHighContrast.ops(),
            LookOps::Agx(grade(1.57, Some(0.9)))
        );
        assert_eq!(
            Look::AgxMediumHighContrast.ops(),
            LookOps::Agx(grade(1.2, None))
        );
    }
}
