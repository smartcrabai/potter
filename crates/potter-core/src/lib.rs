#![recursion_limit = "512"]
#![cfg_attr(
    test,
    expect(
        clippy::float_cmp,
        reason = "unit tests assert exact results for deterministic operations"
    )
)]

mod blender_rng;
pub(crate) mod cache_sequence;
pub(crate) const BUILD_IDENTITY: &str = env!("POTTER_BUILD_IDENTITY");

pub mod catalog;
pub mod cli;
pub mod color;
pub mod commands;
pub mod compositor;
pub mod error;
pub mod eval;
pub mod exchange;
mod float;
pub mod geom;
pub mod graph;
pub mod hash;
pub mod image;
pub mod library;
pub mod mask;
pub mod media;
pub mod model;
pub mod ops;
mod params;
mod physics_support;
pub mod render;
pub mod response;
pub mod schema;
pub mod sequencer;
pub mod shader;
pub mod sim;
pub mod store;
pub mod tracking;
pub mod validate;
