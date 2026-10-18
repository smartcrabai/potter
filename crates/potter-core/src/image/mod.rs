//! Image sampling types and routines.

pub mod sampling;
pub mod storage;
pub use storage::{
    MAX_IMAGE_PIXELS, decode_pixels, decode_png_dimensions, encode_pixels, load_image_data,
    load_image_data_with_staged,
};

pub use sampling::{
    ImageData, ImageExtension, ImageInterpolation, ImageStorage, ImageTileData, Precision, sample,
    sample_encoded, sample_with_extension, sample_with_extension_alpha, sample_with_interpolation,
};
