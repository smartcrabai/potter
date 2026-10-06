use glam::{DMat4, DQuat, DVec3};
use serde::{Deserialize, Serialize};

use crate::error::{ErrorCode, PotError, Result};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Transform {
    #[serde(default = "zero3")]
    pub translation: [f64; 3],
    #[serde(default = "identity_quat", deserialize_with = "deserialize_rotation")]
    pub rotation: [f64; 4],
    #[serde(default = "one3")]
    pub scale: [f64; 3],
    #[serde(default = "default_rotation_mode")]
    pub rotation_mode: String,
}

impl Default for Transform {
    fn default() -> Self {
        Self {
            translation: zero3(),
            rotation: identity_quat(),
            scale: one3(),
            rotation_mode: default_rotation_mode(),
        }
    }
}

impl Transform {
    pub fn from_rotation_quat(
        translation: [f64; 3],
        rotation: [f64; 4],
        scale: [f64; 3],
    ) -> Result<Self> {
        let normalized = normalize_rotation(rotation)?;
        Ok(Self {
            translation,
            rotation: normalized,
            scale,
            rotation_mode: default_rotation_mode(),
        })
    }

    #[must_use]
    pub fn from_rotation_deg(
        translation: [f64; 3],
        rotation_deg: [f64; 3],
        scale: [f64; 3],
    ) -> Self {
        let x = DQuat::from_rotation_x(rotation_deg[0].to_radians());
        let y = DQuat::from_rotation_y(rotation_deg[1].to_radians());
        let z = DQuat::from_rotation_z(rotation_deg[2].to_radians());
        let q = z * y * x;
        Self {
            translation,
            rotation: canonicalize_quaternion([q.x, q.y, q.z, q.w]),
            scale,
            rotation_mode: default_rotation_mode(),
        }
    }

    #[must_use]
    pub fn matrix(&self) -> DMat4 {
        let translation = DMat4::from_translation(DVec3::from_array(self.translation));
        let rotation = DMat4::from_quat(DQuat::from_xyzw(
            self.rotation[0],
            self.rotation[1],
            self.rotation[2],
            self.rotation[3],
        ));
        let scale = DMat4::from_scale(DVec3::from_array(self.scale));
        translation * rotation * scale
    }

    #[must_use]
    pub fn rotation_quat(&self) -> DQuat {
        DQuat::from_xyzw(
            self.rotation[0],
            self.rotation[1],
            self.rotation[2],
            self.rotation[3],
        )
    }
}

pub fn normalize_rotation(rotation: [f64; 4]) -> Result<[f64; 4]> {
    if rotation.iter().any(|value| !value.is_finite()) {
        return Err(PotError::new(
            ErrorCode::InvalidOperation,
            "rotation quaternion must be finite",
        ));
    }
    let largest = rotation
        .iter()
        .map(|value| value.abs())
        .fold(0.0_f64, f64::max);
    if largest == 0.0 {
        return Err(PotError::new(
            ErrorCode::InvalidOperation,
            "rotation quaternion must be non-zero",
        ));
    }
    let scaled = rotation.map(|value| value / largest);
    let length = scaled.iter().map(|value| value * value).sum::<f64>().sqrt();
    // Preserve already-unit quaternions; re-normalizing can change serialized bits and scene hashes.
    if (largest * length - 1.0).abs() <= 8.0 * f64::EPSILON {
        return Ok(canonicalize_quaternion(rotation));
    }
    Ok(canonicalize_quaternion(scaled.map(|value| value / length)))
}

fn deserialize_rotation<'de, D>(deserializer: D) -> std::result::Result<[f64; 4], D::Error>
where
    D: serde::Deserializer<'de>,
{
    let rotation = <[f64; 4]>::deserialize(deserializer)?;
    normalize_rotation(rotation).map_err(serde::de::Error::custom)
}

#[must_use]
pub fn canonicalize_quaternion(mut rotation: [f64; 4]) -> [f64; 4] {
    let flip = if rotation[3] < 0.0 {
        true
    } else if rotation[3] == 0.0 {
        rotation[..3]
            .iter()
            .copied()
            .find(|component| *component != 0.0)
            .is_some_and(|first| first < 0.0)
    } else {
        false
    };
    if flip {
        for component in &mut rotation {
            *component = -*component;
        }
    }
    rotation
}

fn zero3() -> [f64; 3] {
    [0.0; 3]
}

fn one3() -> [f64; 3] {
    [1.0; 3]
}

fn identity_quat() -> [f64; 4] {
    [0.0, 0.0, 0.0, 1.0]
}

fn default_rotation_mode() -> String {
    "quaternion".to_owned()
}

#[cfg(test)]
mod tests {
    #![expect(clippy::unwrap_used, reason = "tests")]

    use super::{Transform, canonicalize_quaternion, normalize_rotation};
    use glam::{DMat4, DQuat, DVec3};
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn rotation_deg_survives_serialization_without_changing_bits(
            angles in prop::array::uniform3(-180.0_f64..180.0_f64),
        ) {
            let transform =
                Transform::from_rotation_deg([4.0, -4.0, 3.0], angles, [1.0; 3]);
            let value = serde_json::to_value(&transform).unwrap();
            let decoded: Transform = serde_json::from_value(value).unwrap();
            prop_assert_eq!(decoded.rotation, transform.rotation);
        }

        #[test]
        fn trs_matrix_matches_glam_reference(
            translation in prop::array::uniform3(-1.0e4_f64..1.0e4),
            angles in prop::array::uniform3(-180.0_f64..180.0),
            scale in prop::array::uniform3(-10.0_f64..10.0),
        ) {
            let transform = Transform::from_rotation_deg(translation, angles, scale);
            let reference = DMat4::from_scale_rotation_translation(
                DVec3::from_array(scale),
                transform.rotation_quat(),
                DVec3::from_array(translation),
            );
            for (actual, expected) in transform
                .matrix()
                .to_cols_array()
                .into_iter()
                .zip(reference.to_cols_array())
            {
                prop_assert!((actual - expected).abs() <= 1.0e-11 * expected.abs().max(1.0));
            }
        }
    }

    proptest! {
        #[test]
        fn normalized_quaternions_are_unit_and_canonical(values in prop::array::uniform4(-1.0e6_f64..1.0e6_f64)) {
            let norm = values.iter().map(|value| value * value).sum::<f64>().sqrt();
            prop_assume!(norm > f64::EPSILON);
            let q = normalize_rotation(values).unwrap();
            let output_norm = q.iter().map(|value| value * value).sum::<f64>().sqrt();
            prop_assert!((output_norm - 1.0).abs() < 1.0e-12);
            prop_assert_eq!(normalize_rotation(q).unwrap(), q);
            prop_assert!(q[3] > 0.0 || (q[3] == 0.0 && q[..3].iter().copied().find(|v| *v != 0.0).is_some_and(|v| v > 0.0)));
        }
    }

    #[test]
    fn normalization_is_stable_for_large_finite_quaternions() {
        let q = normalize_rotation([1.0e308; 4]).unwrap();
        let norm = q.iter().map(|value| value * value).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1.0e-12);
    }

    #[test]
    fn deserialized_quaternions_are_normalized_and_have_canonical_sign() {
        let transform: Transform = serde_json::from_str(r#"{"rotation":[0,-2,0,0]}"#).unwrap();
        assert_eq!(transform.rotation, [0.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn rotation_deg_is_rz_ry_rx_and_matrix_is_trs() {
        let transform =
            Transform::from_rotation_deg([1.0, 2.0, 3.0], [0.0, 0.0, 90.0], [2.0, 1.0, 1.0]);
        let point = transform.matrix().transform_point3(DVec3::X);
        assert!((point - DVec3::new(1.0, 4.0, 3.0)).length() < 1.0e-12);
    }

    #[test]
    fn default_transform_keeps_contract_translation_rotation_scale_and_mode() {
        let transform = Transform::default();
        assert_eq!(transform.translation, [0.0; 3]);
        assert_eq!(transform.rotation, [0.0, 0.0, 0.0, 1.0]);
        assert_eq!(transform.scale, [1.0; 3]);
        assert_eq!(transform.rotation_mode, "quaternion");
    }

    #[test]
    fn quaternion_constructor_preserves_trs_fields_and_normalizes_rotation() {
        let transform =
            Transform::from_rotation_quat([4.0, 5.0, 6.0], [0.0, 0.0, 2.0, 0.0], [2.0, 3.0, 4.0])
                .unwrap();
        assert_eq!(transform.translation, [4.0, 5.0, 6.0]);
        assert_eq!(transform.rotation, [0.0, 0.0, 1.0, 0.0]);
        assert_eq!(transform.scale, [2.0, 3.0, 4.0]);
        assert_eq!(transform.rotation_mode, "quaternion");
    }

    #[test]
    fn degree_rotation_matches_non_commuting_glam_z_y_x_reference() {
        let degrees = [31.0_f64, -27.0, 43.0];
        let expected = DQuat::from_rotation_z(degrees[2].to_radians())
            * DQuat::from_rotation_y(degrees[1].to_radians())
            * DQuat::from_rotation_x(degrees[0].to_radians());
        let actual =
            Transform::from_rotation_deg([4.0, -4.0, 3.0], degrees, [1.0; 3]).rotation_quat();
        assert!((actual.dot(expected).abs() - 1.0).abs() < 1.0e-12);
    }

    #[test]
    fn quaternion_sign_canonicalization_uses_the_first_nonzero_component() {
        assert_eq!(
            canonicalize_quaternion([0.0, -1.0, 2.0, 0.0]),
            [0.0, 1.0, -2.0, 0.0]
        );
        assert_eq!(
            canonicalize_quaternion([0.0, 0.0, -1.0, 0.0]),
            [0.0, 0.0, 1.0, 0.0]
        );
        assert_eq!(
            canonicalize_quaternion([0.0, 0.0, 1.0, 0.0]),
            [0.0, 0.0, 1.0, 0.0]
        );
        assert_eq!(
            canonicalize_quaternion([1.0, 2.0, 3.0, -4.0]),
            [-1.0, -2.0, -3.0, 4.0]
        );
    }
}
