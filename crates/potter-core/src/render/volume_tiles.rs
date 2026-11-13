//! Cycles' CPU render-tile layout (`session/session.cpp` and `session/tile.cpp`).

const IMAGE_TILE_SIZE: u64 = 128;
const MAX_TILE_SIZE: u64 = 8192;
const ADAPTIVE_OVERSCAN: u64 = 4;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct VolumeTile {
    pub buffer_x: u32,
    pub buffer_y: u32,
    pub buffer_width: u32,
    pub buffer_height: u32,
    pub window_x: u32,
    pub window_y: u32,
    pub window_width: u32,
    pub window_height: u32,
}

/// Lazy row-major iterator for Cycles' tile windows and their source buffers.
pub(super) struct VolumeTiles {
    width: u32,
    height: u32,
    tile_width: u64,
    tile_height: u64,
    tiles_x: u64,
    next_index: u64,
    num_tiles: u64,
    overscan: u64,
}

impl VolumeTiles {
    pub(super) fn new(width: u32, height: u32, tile_size: u32, adaptive: bool) -> Self {
        let actual_tile_size = compute_render_tile_size(tile_size);
        let image_area = u64::from(width) * u64::from(height);
        let tile_area = actual_tile_size * actual_tile_size;
        let whole_image = tile_area >= image_area
            && u64::from(width) <= MAX_TILE_SIZE
            && u64::from(height) <= MAX_TILE_SIZE;

        let (tile_width, tile_height) = if whole_image {
            (u64::from(width), u64::from(height))
        } else {
            (actual_tile_size, actual_tile_size)
        };
        let tiles_x = ceil_div(u64::from(width), tile_width);
        let tiles_y = ceil_div(u64::from(height), tile_height);
        let num_tiles = tiles_x * tiles_y;
        let overscan = if adaptive && num_tiles > 1 {
            ADAPTIVE_OVERSCAN
        } else {
            0
        };

        Self {
            width,
            height,
            tile_width,
            tile_height,
            tiles_x,
            next_index: 0,
            num_tiles,
            overscan,
        }
    }
}

impl Iterator for VolumeTiles {
    type Item = VolumeTile;

    fn next(&mut self) -> Option<Self::Item> {
        if self.next_index >= self.num_tiles {
            return None;
        }

        let tile_y = self.next_index / self.tiles_x;
        let tile_x = self.next_index - tile_y * self.tiles_x;
        self.next_index += 1;

        let window_x = tile_x * self.tile_width;
        let window_y = tile_y * self.tile_height;
        let window_width = self.tile_width.min(u64::from(self.width) - window_x);
        let window_height = self.tile_height.min(u64::from(self.height) - window_y);
        let buffer_x = window_x.saturating_sub(self.overscan);
        let buffer_y = window_y.saturating_sub(self.overscan);
        let local_window_x = window_x - buffer_x;
        let local_window_y = window_y - buffer_y;
        let buffer_width =
            (u64::from(self.width) - buffer_x).min(local_window_x + window_width + self.overscan);
        let buffer_height =
            (u64::from(self.height) - buffer_y).min(local_window_y + window_height + self.overscan);

        Some(VolumeTile {
            buffer_x: buffer_x as u32,
            buffer_y: buffer_y as u32,
            buffer_width: buffer_width as u32,
            buffer_height: buffer_height as u32,
            window_x: local_window_x as u32,
            window_y: local_window_y as u32,
            window_width: window_width as u32,
            window_height: window_height as u32,
        })
    }
}

fn compute_render_tile_size(suggested_tile_size: u32) -> u64 {
    let suggested = u64::from(suggested_tile_size);
    let aligned = if suggested <= IMAGE_TILE_SIZE {
        suggested
    } else {
        suggested.div_ceil(IMAGE_TILE_SIZE) * IMAGE_TILE_SIZE
    };
    aligned.min(MAX_TILE_SIZE)
}

fn ceil_div(value: u64, divisor: u64) -> u64 {
    if divisor == 0 {
        0
    } else {
        value.div_ceil(divisor)
    }
}

#[cfg(test)]
mod tests {
    use super::{VolumeTile, VolumeTiles};

    fn collect_tiles(width: u32, height: u32, tile_size: u32, adaptive: bool) -> Vec<VolumeTile> {
        VolumeTiles::new(width, height, tile_size, adaptive).collect()
    }

    #[test]
    fn small_image_uses_one_whole_image_tile_at_area_threshold() {
        assert_eq!(
            collect_tiles(100, 100, 128, true),
            [VolumeTile {
                buffer_x: 0,
                buffer_y: 0,
                buffer_width: 100,
                buffer_height: 100,
                window_x: 0,
                window_y: 0,
                window_width: 100,
                window_height: 100,
            }]
        );
        assert_eq!(collect_tiles(128, 128, 128, true).len(), 1);
        assert_eq!(collect_tiles(128, 129, 128, false).len(), 2);
    }

    #[test]
    fn adaptive_3840_by_2160_tiles_have_exact_halo_and_crops() {
        let tiles = collect_tiles(3840, 2160, 2048, true);
        assert_eq!(
            tiles,
            [
                VolumeTile {
                    buffer_x: 0,
                    buffer_y: 0,
                    buffer_width: 2052,
                    buffer_height: 2052,
                    window_x: 0,
                    window_y: 0,
                    window_width: 2048,
                    window_height: 2048,
                },
                VolumeTile {
                    buffer_x: 2044,
                    buffer_y: 0,
                    buffer_width: 1796,
                    buffer_height: 2052,
                    window_x: 4,
                    window_y: 0,
                    window_width: 1792,
                    window_height: 2048,
                },
                VolumeTile {
                    buffer_x: 0,
                    buffer_y: 2044,
                    buffer_width: 2052,
                    buffer_height: 116,
                    window_x: 0,
                    window_y: 4,
                    window_width: 2048,
                    window_height: 112,
                },
                VolumeTile {
                    buffer_x: 2044,
                    buffer_y: 2044,
                    buffer_width: 1796,
                    buffer_height: 116,
                    window_x: 4,
                    window_y: 4,
                    window_width: 1792,
                    window_height: 112,
                },
            ]
        );
        assert_crops_partition_image(3840, 2160, &tiles);
    }

    #[test]
    fn tile_size_matches_cycles_alignment_and_cap() {
        let tiles = collect_tiles(17, 17, 8, false);
        assert_eq!(tiles[0].window_width, 8);
        assert_eq!(tiles[0].window_height, 8);

        let tiles = collect_tiles(300, 300, 129, false);
        assert_eq!(tiles[0].window_width, 256);
        assert_eq!(tiles[0].window_height, 256);

        let tiles = collect_tiles(9001, 3, 8192, true);
        assert_eq!(tiles.len(), 2);
        assert_eq!(tiles[0].window_width, 8192);
        assert_eq!(tiles[1].buffer_x, 8188);
        assert_eq!(tiles[1].window_x, 4);
        assert_eq!(tiles[1].window_width, 809);
        assert_crops_partition_image(9001, 3, &tiles);
    }

    #[test]
    fn skinny_images_obey_bounds_and_zero_dimensions_are_empty() {
        let tiles = collect_tiles(9001, 3, 2048, true);
        assert_eq!(tiles.len(), 5);
        for tile in &tiles {
            assert!(u64::from(tile.buffer_x) + u64::from(tile.buffer_width) <= 9001);
            assert!(u64::from(tile.buffer_y) + u64::from(tile.buffer_height) <= 3);
        }
        assert_crops_partition_image(9001, 3, &tiles);

        assert_eq!(collect_tiles(0, 100, 128, true), Vec::<VolumeTile>::new());
        assert_eq!(collect_tiles(100, 0, 128, true), Vec::<VolumeTile>::new());
        assert_eq!(collect_tiles(0, 0, 128, true), Vec::<VolumeTile>::new());
    }

    #[test]
    fn u32_sized_images_yield_a_finite_prefix_with_wide_coordinates() -> Result<(), &'static str> {
        let mut tiles = VolumeTiles::new(u32::MAX, u32::MAX, 8192, false);
        let first = tiles.next().ok_or("missing first tile")?;
        let second = tiles.next().ok_or("missing second tile")?;
        let third = tiles.next().ok_or("missing third tile")?;

        assert_eq!(first.buffer_x, 0);
        assert_eq!(first.buffer_y, 0);
        assert_eq!(first.window_width, 8192);
        assert_eq!(second.buffer_x, 8192);
        assert_eq!(third.buffer_x, 16384);
        for tile in [first, second, third] {
            assert!(u32::try_from(u64::from(tile.buffer_x) + u64::from(tile.buffer_width)).is_ok());
            assert!(
                u32::try_from(u64::from(tile.buffer_y) + u64::from(tile.buffer_height)).is_ok()
            );
        }
        Ok(())
    }

    fn assert_crops_partition_image(width: u32, height: u32, tiles: &[VolumeTile]) {
        let mut area = 0_u64;
        for (index, tile) in tiles.iter().enumerate() {
            let left = u64::from(tile.buffer_x) + u64::from(tile.window_x);
            let top = u64::from(tile.buffer_y) + u64::from(tile.window_y);
            let right = left + u64::from(tile.window_width);
            let bottom = top + u64::from(tile.window_height);
            assert!(right <= u64::from(width));
            assert!(bottom <= u64::from(height));
            area += u64::from(tile.window_width) * u64::from(tile.window_height);

            for other in &tiles[index + 1..] {
                let other_left = u64::from(other.buffer_x) + u64::from(other.window_x);
                let other_top = u64::from(other.buffer_y) + u64::from(other.window_y);
                let other_right = other_left + u64::from(other.window_width);
                let other_bottom = other_top + u64::from(other.window_height);
                assert!(
                    right <= other_left
                        || other_right <= left
                        || bottom <= other_top
                        || other_bottom <= top
                );
            }
        }
        assert_eq!(area, u64::from(width) * u64::from(height));
    }
}
