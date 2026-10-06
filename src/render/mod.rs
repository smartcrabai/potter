pub mod raster;

mod bvh;
mod path;
mod pick;
mod preview;
mod scene;
mod sequence;

pub use pick::pick;
pub use preview::{PreviewRequest, render_previews};
pub use sequence::render_sequence;

#[cfg(test)]
mod tests {

    use glam::DVec3;
    use proptest::prelude::*;

    use super::raster::{Camera, Projection, ray_for_pixel};

    proptest! {
        #[test]
        fn pixel_center_ray_round_trips_to_orthographic_pixel(x in 0_u32..128, y in 0_u32..128) {
            let camera = Camera {
                position: DVec3::new(0.0, 0.0, 5.0),
                target: DVec3::ZERO,
                up: DVec3::Y,
                near: 0.01,
                far: 100.0,
                projection: Projection::Orthographic { height: 2.0 },
                shift: [0.0; 2],
                depth_of_field: None,
                stereo_mode: super::raster::StereoMode::None,
                interocular_distance: 0.065,
            };
            let ray = ray_for_pixel(&camera, 128, 128, x, y);
            prop_assert!(ray.is_ok());
            let Ok((origin, direction)) = ray else {
                return Ok(());
            };
            let forward = (camera.target - camera.position).normalize();
            let distance = (camera.target - origin).dot(forward) / direction.dot(forward);
            let world = origin + direction * distance;
            let pixel_x = ((world.x + 1.0) * 64.0 - 0.5).round();
            let pixel_y = ((1.0 - world.y) * 64.0 - 0.5).round();
            prop_assert!((pixel_x - f64::from(x)).abs() < 1.0e-9);
            prop_assert!((pixel_y - f64::from(y)).abs() < 1.0e-9);
        }
    }
}
