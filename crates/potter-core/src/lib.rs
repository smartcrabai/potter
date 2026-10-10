#![recursion_limit = "512"]
#![cfg_attr(
    test,
    expect(
        clippy::float_cmp,
        reason = "unit tests assert exact results for deterministic operations"
    )
)]

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
pub mod render;
pub mod response;
pub mod schema;
pub mod sequencer;
pub mod shader;
pub mod sim;
pub mod store;
pub mod tracking;
pub mod validate;
