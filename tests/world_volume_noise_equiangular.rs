#[path = "common/blender_checked.rs"]
mod blender_checked;
#[path = "common/native_oracle.rs"]
pub mod native_oracle;
#[path = "common/process.rs"]
mod process;
#[path = "common/world_volume_oracle.rs"]
mod world_volume_oracle;

#[test]
fn noise_equiangular_matches_cycles() -> native_oracle::TestResult {
    world_volume_oracle::run_cases(&["noise_equiangular"], 512)
}
