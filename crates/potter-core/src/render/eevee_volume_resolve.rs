//! EEVEE 5.2.2 froxel coordinates used by the volume resolve.
//!
//! Source references (Blender v5.2.2): `draw/engines/eevee/eevee_volume.cc:30-58,153-159` and
//! `draw/engines/eevee/shaders/eevee_volume_lib.bsl.hh:32-74,121-131,223-233`.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum VolumeProjection {
    Perspective,
    Orthographic,
}

/// Depth parameters used to map between normalized froxel Z and signed view-space Z.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct VolumeDepthMapping {
    projection: VolumeProjection,
    depth_near: f32,
    depth_far: f32,
    distribution: f32,
}

impl VolumeDepthMapping {
    /// Builds the EEVEE depth mapping from the signed view-space volume range.
    ///
    /// `volume_near` and `volume_far` are negative view-Z values, as in `eevee_volume.cc`.
    pub(super) fn new(
        projection: VolumeProjection,
        volume_near: f32,
        volume_far: f32,
        volumetric_sample_distribution: f32,
    ) -> Self {
        match projection {
            VolumeProjection::Perspective => {
                let distribution = 4.0 * (1.0 - volumetric_sample_distribution).max(1e-2);
                let depth_near = (volume_far - volume_near * (1.0 / distribution).exp2())
                    / (volume_far - volume_near);
                let depth_far = (1.0 - depth_near) / volume_near;
                Self {
                    projection,
                    depth_near,
                    depth_far,
                    distribution,
                }
            }
            VolumeProjection::Orthographic => Self {
                projection,
                depth_near: volume_near,
                depth_far: volume_far,
                distribution: 0.0,
            },
        }
    }

    /// Converts normalized froxel Z to signed view-space Z.
    pub(super) fn volume_z_to_view_z(self, z: f32) -> f32 {
        match self.projection {
            VolumeProjection::Perspective => {
                ((z / self.distribution).exp2() - self.depth_near) / self.depth_far
            }
            VolumeProjection::Orthographic => {
                self.depth_near + (self.depth_far - self.depth_near) * z
            }
        }
    }

    /// Converts signed view-space Z to normalized froxel Z.
    pub(super) fn view_z_to_volume_z(self, view_z: f32) -> f32 {
        match self.projection {
            VolumeProjection::Perspective => {
                self.distribution * (view_z * self.depth_far + self.depth_near).max(1e-8).log2()
            }
            VolumeProjection::Orthographic => {
                (view_z - self.depth_near) / (self.depth_far - self.depth_near)
            }
        }
    }
}

/// Froxel XY dimensions and screen-to-froxel coordinate scale for a render extent.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct FroxelTextureLayout {
    pub(super) dimensions: [u32; 2],
    pub(super) coord_scale: [f32; 2],
}

impl FroxelTextureLayout {
    /// Computes ceil-divided froxel dimensions and Blender's padded-extent scale.
    pub(super) fn new(width: u32, height: u32, tile_size: u32) -> Self {
        assert!(width > 0 && height > 0, "render extent must be nonzero");
        assert!(tile_size > 0, "froxel tile size must be nonzero");
        let dimensions = [width.div_ceil(tile_size), height.div_ceil(tile_size)];
        let coord_scale = [
            width as f32 / (tile_size as f32 * dimensions[0] as f32),
            height as f32 / (tile_size as f32 * dimensions[1] as f32),
        ];
        Self {
            dimensions,
            coord_scale,
        }
    }
}

/// Maps screen UV to resolve UV following `eevee_volume_lib.bsl.hh:123-130`.
pub(super) fn screen_uv_to_resolve_uv(screen_uv: [f32; 2], coord_scale: [f32; 2]) -> [f32; 2] {
    [screen_uv[0] * coord_scale[0], screen_uv[1] * coord_scale[1]]
}

/// Applies EEVEE's one-slice negative resolve bias from `eevee_volume_lib.bsl.hh:223-233`.
///
/// The integration shader stores accumulated transmittance after each cell's step
/// (`eevee_volume.bsl.hh:341-372`), so this samples the previous slice endpoint. The returned
/// coordinate is not clamped; [`linear_sample_weights`] clamps when deriving actual sample indices.
pub(super) fn resolve_z_with_end_of_slice_bias(z: f32, texture_depth: usize) -> f32 {
    assert!(texture_depth > 0, "froxel texture depth must be nonzero");
    z - 1.0 / texture_depth as f32
}

/// The two texel indices and upper-texel blend weight for normalized linear texture sampling.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct LinearSampleWeights {
    pub(super) lower_index: usize,
    pub(super) upper_index: usize,
    pub(super) upper_weight: f32,
}

/// Computes clamp-to-edge linear-sampler indices and weights for a normalized coordinate.
///
/// Texture texel centers are at `(index + 0.5) / texel_count`, matching normalized GPU sampling.
pub(super) fn linear_sample_weights(
    normalized_coordinate: f32,
    texel_count: usize,
) -> LinearSampleWeights {
    assert!(texel_count > 0, "texture axis size must be nonzero");
    let coordinate = normalized_coordinate.clamp(0.0, 1.0);
    let texel_position = coordinate * texel_count as f32 - 0.5;
    let lower_unclamped = texel_position.floor() as isize;
    let last_index = (texel_count - 1) as isize;
    let lower_index = lower_unclamped.clamp(0, last_index) as usize;
    let upper_index = (lower_unclamped + 1).clamp(0, last_index) as usize;

    LinearSampleWeights {
        lower_index,
        upper_index,
        upper_weight: texel_position - lower_unclamped as f32,
    }
}

#[cfg(test)]
mod tests {
    use super::{
        FroxelTextureLayout, LinearSampleWeights, VolumeDepthMapping, VolumeProjection,
        linear_sample_weights, resolve_z_with_end_of_slice_bias, screen_uv_to_resolve_uv,
    };

    fn assert_close(actual: f32, expected: f32) {
        assert!((actual - expected).abs() < 1e-5, "{actual} != {expected}");
    }

    #[test]
    fn orthographic_depth_mapping_round_trips() {
        let mapping = VolumeDepthMapping::new(VolumeProjection::Orthographic, -0.25, -42.0, 0.5);
        for z in [0.0, 0.2, 0.5, 1.0] {
            assert_close(mapping.view_z_to_volume_z(mapping.volume_z_to_view_z(z)), z);
        }
    }

    #[test]
    fn perspective_depth_mapping_round_trips() {
        let mapping = VolumeDepthMapping::new(VolumeProjection::Perspective, -0.1, -100.0, 0.2);
        for z in [0.0, 0.1, 0.5, 0.9, 1.0] {
            assert_close(mapping.view_z_to_volume_z(mapping.volume_z_to_view_z(z)), z);
        }
    }

    #[test]
    fn padded_xy_extent_uses_ceil_dimensions_and_scale() {
        let layout = FroxelTextureLayout::new(101, 53, 16);
        assert_eq!(layout.dimensions, [7, 4]);
        assert_close(layout.coord_scale[0], 101.0 / (16.0 * 7.0));
        assert_close(layout.coord_scale[1], 53.0 / (16.0 * 4.0));
        assert_eq!(
            screen_uv_to_resolve_uv([1.0, 1.0], layout.coord_scale),
            layout.coord_scale,
        );
    }

    #[test]
    fn linear_sampling_returns_center_and_edge_weights() {
        assert_eq!(
            linear_sample_weights(0.375, 4),
            LinearSampleWeights {
                lower_index: 1,
                upper_index: 2,
                upper_weight: 0.0,
            }
        );
        assert_eq!(
            linear_sample_weights(0.5, 4),
            LinearSampleWeights {
                lower_index: 1,
                upper_index: 2,
                upper_weight: 0.5,
            }
        );
        assert_eq!(
            linear_sample_weights(0.0, 4),
            LinearSampleWeights {
                lower_index: 0,
                upper_index: 0,
                upper_weight: 0.5,
            }
        );
        assert_eq!(
            linear_sample_weights(1.0, 4),
            LinearSampleWeights {
                lower_index: 3,
                upper_index: 3,
                upper_weight: 0.5,
            }
        );
        assert_eq!(
            linear_sample_weights(-1.0, 4).lower_index,
            linear_sample_weights(0.0, 4).lower_index,
        );
        assert_eq!(
            linear_sample_weights(2.0, 4).upper_index,
            linear_sample_weights(1.0, 4).upper_index,
        );
    }

    #[test]
    fn resolve_bias_moves_to_previous_slice_endpoint() {
        assert_close(resolve_z_with_end_of_slice_bias(0.5, 4), 0.5 - 1.0 / 4.0);
    }
}
