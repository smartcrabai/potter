#[path = "common/blender_checked.rs"]
mod blender_checked;
#[path = "common/native_oracle.rs"]
pub mod native_oracle;
#[path = "common/process.rs"]
mod process;
#[path = "common/world_volume_oracle.rs"]
mod world_volume_oracle;

#[test]
fn world_override_spatial_matches_cycles() -> native_oracle::TestResult {
    world_volume_oracle::run_cases(&["world_override_spatial"], 512)
}
