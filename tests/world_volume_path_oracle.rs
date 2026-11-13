#[path = "common/blender_checked.rs"]
mod blender_checked;
#[path = "common/native_oracle.rs"]
pub mod native_oracle;
#[path = "common/process.rs"]
mod process;
#[path = "common/world_volume_oracle.rs"]
mod world_volume_oracle;

#[test]
fn homogeneous_world_volume_matches_cycles_with_view_layer_override() -> native_oracle::TestResult {
    world_volume_oracle::run_cases(
        &[
            "scene_world_no_volume",
            "scene_world_absorption_only",
            "scene_world_scattering_only",
            "scene_world_distance",
            "scene_world",
            "view_layer_override",
        ],
        4096,
    )
}
