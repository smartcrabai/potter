//! Bounds-checked reader for scalar and three-component `OpenVDB` grids used by Blender.
//!
//! `OpenVDB`'s tree remains sparse: leaf voxels and non-background tiles are stored as
//! entries, while background values are implicit. The reader intentionally supports
//! the standard `Tree_*_5_4_3` hierarchy and returns typed errors for other variants.

use std::{
    collections::HashMap,
    io::Read,
    sync::{Arc, LazyLock, Mutex},
};

use glam::{DMat4, DVec3, IVec3};
use serde_json::json;

use crate::error::{ErrorCode, PotError, Result};

const VDB_MAGIC: u32 = 0x5644_4220;
const MIN_FILE_VERSION: u32 = 222;
const MAX_FILE_VERSION: u32 = 225;
const MAX_FILE_BYTES: usize = 1 << 30;
const MAX_GRID_COUNT: usize = 64;
const MAX_METADATA_COUNT: usize = 16_384;
const MAX_STRING_BYTES: usize = 1 << 20;
const MAX_NODE_COUNT: usize = 100_000;
const MAX_VOXELS: usize = 1_000_000;
const MAX_COMPRESSED_BYTES: usize = 1 << 30;
const NODE_MASK_COMPRESSION_VERSION: u32 = 222;
const BLOSC_COMPRESSION: u32 = 0x4;
const ZIP_COMPRESSION: u32 = 0x1;
const ACTIVE_MASK_COMPRESSION: u32 = 0x2;
const BLOSC_HEADER_SIZE: usize = 16;
const BLOSC_DOSHUFFLE: u8 = 0x1;
const BLOSC_MEMCPYED: u8 = 0x2;
const BLOSC_DOBITSHUFFLE: u8 = 0x4;
const BLOSC_DONT_SPLIT: u8 = 0x10;
const BLOSC_MAX_SPLITS: usize = 16;
const BLOSC_MIN_ZSTD_WINDOW_BYTES: u64 = 1 << 20;

static VDB_CACHE: LazyLock<Mutex<HashMap<String, Arc<VdbVolume>>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// Scalar or three-component values stored by a VDB tree.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum VdbValue {
    /// A scalar floating-point value.
    Float(f32),
    /// A three-component floating-point value.
    Vec3([f32; 3]),
}

impl VdbValue {
    fn magnitude(self) -> f32 {
        match self {
            Self::Float(value) => value,
            Self::Vec3(value) => value[0].hypot(value[1]).hypot(value[2]),
        }
    }

    fn component(self, component: usize) -> Option<f32> {
        match self {
            Self::Float(value) if component == 0 => Some(value),
            Self::Vec3(value) => value.get(component).copied(),
            Self::Float(_) => None,
        }
    }
}

/// `OpenVDB` index-to-world transform supported by the reader.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VdbTransform {
    /// Matrix mapping index coordinates to world coordinates.
    pub index_to_world: DMat4,
    /// Matrix mapping world coordinates to index coordinates.
    pub world_to_index: DMat4,
    /// Serialized map name.
    pub map_type: &'static str,
}

/// A sparse VDB tree containing non-background voxels and tiles.
#[derive(Clone, Debug, PartialEq)]
pub struct VdbTree {
    /// Tree background value; vector backgrounds are represented by their magnitude.
    pub background: f32,
    background_value: VdbValue,
    values: HashMap<IVec3, VdbValue>,
    tiles: HashMap<(IVec3, i32), VdbTile>,
    /// Tight bounds of active voxels or active tiles in index space.
    pub active_bbox: Option<([i32; 3], [i32; 3])>,
}

/// An `OpenVDB` tile, whose value is constant over a power-of-two cube.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VdbTile {
    /// Inclusive index-space minimum corner.
    pub min: [i32; 3],
    /// Inclusive index-space maximum corner.
    pub max: [i32; 3],
    /// Stored tile value.
    pub value: VdbValue,
    /// Whether the tile is active.
    pub active: bool,
}

/// One decoded `OpenVDB` grid.
#[derive(Clone, Debug, PartialEq)]
pub struct VdbGrid {
    /// Grid name as written in the file.
    pub name: String,
    /// Grid class metadata, when present.
    pub class: Option<String>,
    /// `OpenVDB` value type string.
    pub value_type: String,
    /// Grid index/world transform.
    pub transform: VdbTransform,
    /// Sparse tree and background.
    pub tree: Arc<VdbTree>,
    /// Active index-space bounding box.
    pub active_bbox: Option<([i32; 3], [i32; 3])>,
}

impl VdbGrid {
    /// Return the value at an integer index, with the background outside active data.
    #[must_use]
    pub fn sample_index(&self, ijk: [i32; 3]) -> f32 {
        self.sample_value(IVec3::from_array(ijk)).magnitude()
    }

    /// Return one scalar component at an integer index; scalar grids have component 0.
    #[must_use]
    pub fn sample_index_component(&self, ijk: [i32; 3], component: usize) -> Option<f32> {
        self.sample_value(IVec3::from_array(ijk))
            .component(component)
    }

    /// Trilinearly sample the grid in world coordinates.
    #[must_use]
    pub fn sample_world(&self, position: DVec3) -> f64 {
        if !position.is_finite() {
            return f64::from(self.tree.background);
        }
        let index = self.transform.world_to_index.transform_point3(position);
        if !index.is_finite() {
            return f64::from(self.tree.background);
        }
        let lower = [index.x.floor(), index.y.floor(), index.z.floor()];
        if lower
            .iter()
            .any(|value| *value < f64::from(i32::MIN) || *value >= f64::from(i32::MAX))
        {
            return f64::from(self.tree.background);
        }
        let base = lower.map(|value| value as i32);
        let fraction = [index.x - lower[0], index.y - lower[1], index.z - lower[2]];
        let mut result = 0.0_f64;
        for z in 0..=1 {
            for y in 0..=1 {
                for x in 0..=1 {
                    let point = [base[0] + x, base[1] + y, base[2] + z];
                    let weight = (if x == 0 {
                        1.0 - fraction[0]
                    } else {
                        fraction[0]
                    }) * (if y == 0 {
                        1.0 - fraction[1]
                    } else {
                        fraction[1]
                    }) * (if z == 0 {
                        1.0 - fraction[2]
                    } else {
                        fraction[2]
                    });
                    result += f64::from(self.sample_index(point)) * weight;
                }
            }
        }
        result
    }

    /// Return the world-space AABB enclosing the active index-space bounds.
    #[must_use]
    pub fn active_bbox_world(&self) -> Option<(DVec3, DVec3)> {
        let (minimum, maximum) = self.active_bbox?;
        let mut world_min = DVec3::splat(f64::INFINITY);
        let mut world_max = DVec3::splat(f64::NEG_INFINITY);
        for corner in 0..8 {
            let index = DVec3::new(
                f64::from(if corner & 1 == 0 {
                    minimum[0]
                } else {
                    maximum[0]
                }),
                f64::from(if corner & 2 == 0 {
                    minimum[1]
                } else {
                    maximum[1]
                }),
                f64::from(if corner & 4 == 0 {
                    minimum[2]
                } else {
                    maximum[2]
                }),
            );
            let world = self.transform.index_to_world.transform_point3(index);
            if !world.is_finite() {
                return None;
            }
            world_min = world_min.min(world);
            world_max = world_max.max(world);
        }
        Some((world_min, world_max))
    }

    /// Return index-axis voxel spacing in grid-local world coordinates.
    #[must_use]
    pub fn voxel_size(&self) -> DVec3 {
        DVec3::new(
            self.transform
                .index_to_world
                .transform_vector3(DVec3::X)
                .length(),
            self.transform
                .index_to_world
                .transform_vector3(DVec3::Y)
                .length(),
            self.transform
                .index_to_world
                .transform_vector3(DVec3::Z)
                .length(),
        )
    }

    fn sample_value(&self, point: IVec3) -> VdbValue {
        if let Some(value) = self.tree.values.get(&point) {
            return *value;
        }
        for width in [8, 128, 4096] {
            let origin = IVec3::new(
                point.x.div_euclid(width) * width,
                point.y.div_euclid(width) * width,
                point.z.div_euclid(width) * width,
            );
            if let Some(tile) = self.tree.tiles.get(&(origin, width)) {
                return tile.value;
            }
        }
        self.tree.background_value
    }
}

/// A decoded `OpenVDB` archive.
#[derive(Clone, Debug, PartialEq)]
pub struct VdbVolume {
    /// Decoded grids, in archive order.
    pub grids: Vec<VdbGrid>,
    /// Archive UUID.
    pub uuid: String,
    /// `OpenVDB` library version recorded in the file.
    pub library_version: (u32, u32),
}

impl VdbVolume {
    /// Select Blender's density grid, otherwise the first scalar grid or first grid.
    #[must_use]
    pub fn density_grid(&self) -> Option<&VdbGrid> {
        self.grids
            .iter()
            .find(|grid| {
                grid.name == "density" && grid.value_type.split('_').nth(1) == Some("float")
            })
            .or_else(|| {
                self.grids
                    .iter()
                    .find(|grid| grid.value_type.split('_').nth(1) == Some("float"))
            })
            .or_else(|| self.grids.first())
    }
    /// Decode an `OpenVDB` archive from bytes.
    ///
    /// # Errors
    ///
    /// Returns typed import, unsupported-feature, or resource-budget errors for invalid
    /// archives, unsupported maps/types/compression, and oversized topology.
    pub fn read(bytes: &[u8]) -> Result<Self> {
        if bytes.len() > MAX_FILE_BYTES {
            return Err(limit_error("VDB archive exceeds the input byte budget"));
        }
        Parser::new(bytes).read_archive()
    }

    /// Decode an archive once per content hash and share the immutable result in process.
    ///
    /// # Errors
    ///
    /// Returns the same errors as [`VdbVolume::read`].
    pub fn read_cached(bytes: &[u8], content_hash: &str) -> Result<Arc<Self>> {
        let mut cache = VDB_CACHE.lock().map_err(|_| {
            PotError::with_details(
                ErrorCode::InternalError,
                "VDB content cache is poisoned",
                json!({"feature_id":"volume.openvdb_evaluation.cache"}),
            )
        })?;
        if let Some(volume) = cache.get(content_hash) {
            return Ok(Arc::clone(volume));
        }
        let decoded = Arc::new(Self::read(bytes)?);
        if cache.len() >= 8
            && let Some(oldest_key) = cache.keys().next().cloned()
        {
            cache.remove(&oldest_key);
        }
        cache.insert(content_hash.to_owned(), Arc::clone(&decoded));
        Ok(decoded)
    }
}

#[derive(Clone, Debug)]
struct Descriptor {
    name: String,
    grid_type: String,
    instance_parent: String,
    grid_pos: usize,
    block_pos: usize,
    end_pos: usize,
}

#[derive(Clone, Debug)]
struct GridMetadata {
    compression: u32,
    class: Option<String>,
    half_float: bool,
}

#[derive(Clone, Debug)]
struct LeafDescriptor {
    origin: [i32; 3],
    active: Vec<bool>,
}

struct Parser<'a> {
    cursor: Cursor<'a>,
    version: u32,
    library_version: (u32, u32),
    uuid: String,
    global_compression: u32,
    total_voxels: usize,
    total_nodes: usize,
    total_tiles: usize,
}

impl<'a> Parser<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self {
            cursor: Cursor::new(bytes),
            version: 0,
            library_version: (0, 0),
            uuid: String::new(),
            global_compression: 0,
            total_voxels: 0,
            total_nodes: 0,
            total_tiles: 0,
        }
    }

    fn read_archive(mut self) -> Result<VdbVolume> {
        self.read_header()?;
        let descriptors = self.read_descriptors()?;
        let mut grids = Vec::with_capacity(descriptors.len());
        let mut instances = Vec::new();
        for descriptor in descriptors {
            let (metadata, transform, after_transform) = self.read_grid_header(&descriptor)?;
            let value_type = grid_value_type(&descriptor.grid_type)?;
            if descriptor.instance_parent.is_empty() {
                let grid = self.read_grid_tree(
                    &descriptor,
                    metadata,
                    &transform,
                    value_type,
                    after_transform,
                )?;
                grids.push(grid);
            } else {
                instances.push((descriptor, metadata, transform, value_type));
            }
        }
        for (descriptor, metadata, transform, value_type) in instances {
            let parent = grids
                .iter()
                .find(|grid| grid.name == descriptor.instance_parent)
                .ok_or_else(|| import_error("VDB grid instance refers to a missing parent"))?;
            if grid_value_type(&parent.value_type)? != value_type {
                return Err(import_error(
                    "VDB grid instance type differs from its parent",
                ));
            }
            grids.push(VdbGrid {
                name: descriptor.name,
                class: metadata.class,
                value_type: descriptor.grid_type,
                transform,
                tree: Arc::clone(&parent.tree),
                active_bbox: parent.active_bbox,
            });
        }
        Ok(VdbVolume {
            grids,
            uuid: self.uuid,
            library_version: self.library_version,
        })
    }

    fn read_header(&mut self) -> Result<()> {
        if self.cursor.read_u32()? != VDB_MAGIC {
            return Err(import_error("file does not have the OpenVDB magic number"));
        }
        self.cursor.skip(4)?;
        self.version = self.cursor.read_u32()?;
        if !(MIN_FILE_VERSION..=MAX_FILE_VERSION).contains(&self.version) {
            return Err(PotError::with_details(
                ErrorCode::UnsupportedVersion,
                format!("OpenVDB file version {} is unsupported", self.version),
                json!({"feature_id":"volume.openvdb_evaluation.file_version","file_version":self.version}),
            ));
        }
        self.library_version = (self.cursor.read_u32()?, self.cursor.read_u32()?);
        let has_offsets = self.cursor.read_u8()?;
        if has_offsets != 1 {
            return Err(import_error("OpenVDB archive has no grid offsets"));
        }
        if (220..NODE_MASK_COMPRESSION_VERSION).contains(&self.version) {
            self.global_compression = if self.cursor.read_u8()? == 0 {
                ZIP_COMPRESSION
            } else {
                0
            };
        } else if self.version < 223 {
            self.global_compression = ZIP_COMPRESSION | ACTIVE_MASK_COMPRESSION;
        } else {
            self.global_compression = BLOSC_COMPRESSION | ACTIVE_MASK_COMPRESSION;
        }
        let uuid = self.cursor.read_string_fixed(36)?;
        self.uuid = uuid;
        self.read_metadata()?;
        Ok(())
    }

    fn read_descriptors(&mut self) -> Result<Vec<Descriptor>> {
        let grid_count = usize::try_from(self.cursor.read_u32()?)
            .map_err(|_| import_error("VDB grid count is outside platform limits"))?;
        if grid_count > MAX_GRID_COUNT {
            return Err(limit_error("VDB archive contains too many grids"));
        }
        let mut descriptors = Vec::with_capacity(grid_count);
        for _ in 0..grid_count {
            let name = self.cursor.read_string()?;
            let grid_type = self.cursor.read_string()?;
            let instance_parent = if self.version >= 216 {
                self.cursor.read_string()?
            } else {
                String::new()
            };
            let grid_pos = self.cursor.read_offset()?;
            let block_pos = self.cursor.read_offset()?;
            let end_pos = self.cursor.read_offset()?;
            if grid_pos > block_pos || block_pos > end_pos || end_pos > self.cursor.bytes.len() {
                return Err(import_error("VDB grid offsets are out of range"));
            }
            descriptors.push(Descriptor {
                name,
                grid_type,
                instance_parent,
                grid_pos,
                block_pos,
                end_pos,
            });
            // OpenVDB stores each descriptor adjacent to its grid data; the next
            // descriptor begins at this grid's end offset.
            self.cursor.position = end_pos;
        }
        Ok(descriptors)
    }

    fn read_metadata(&mut self) -> Result<HashMap<String, MetadataValue>> {
        let count = usize::try_from(self.cursor.read_u32()?)
            .map_err(|_| import_error("VDB metadata count is outside platform limits"))?;
        if count > MAX_METADATA_COUNT {
            return Err(limit_error("VDB metadata contains too many entries"));
        }
        let mut result = HashMap::with_capacity(count);
        for _ in 0..count {
            let name = self.cursor.read_string()?;
            let metadata_type = self.cursor.read_string()?;
            let length = usize::try_from(self.cursor.read_u32()?)
                .map_err(|_| import_error("VDB metadata size is outside platform limits"))?;
            if length > MAX_FILE_BYTES {
                return Err(limit_error("VDB metadata entry exceeds the byte budget"));
            }
            let bytes = self.cursor.take(length)?;
            let value = match metadata_type.as_str() {
                "string" => Some(MetadataValue::String(
                    std::str::from_utf8(bytes)
                        .map_err(|_| import_error("VDB string metadata is not UTF-8"))?
                        .to_owned(),
                )),
                "bool" if !bytes.is_empty() => Some(MetadataValue::Bool(bytes[0] != 0)),
                _ => None,
            };
            if let Some(value) = value {
                result.insert(name, value);
            }
        }
        Ok(result)
    }

    fn read_grid_header(
        &mut self,
        descriptor: &Descriptor,
    ) -> Result<(GridMetadata, VdbTransform, usize)> {
        self.cursor.position = descriptor.grid_pos;
        let compression = if self.version >= NODE_MASK_COMPRESSION_VERSION {
            self.cursor.read_u32()?
        } else {
            self.global_compression
        };
        validate_compression(compression)?;
        let metadata = self.read_metadata()?;
        let class = match metadata.get("class") {
            Some(MetadataValue::String(value)) => Some(value.clone()),
            _ => None,
        };
        let half_float = descriptor.grid_type.ends_with("_HalfFloat")
            || matches!(
                metadata.get("is_saved_as_half_float"),
                Some(MetadataValue::Bool(true))
            );
        let transform = self.read_transform()?;
        let after_transform = self.cursor.position;
        Ok((
            GridMetadata {
                compression,
                class,
                half_float,
            },
            transform,
            after_transform,
        ))
    }

    fn read_transform(&mut self) -> Result<VdbTransform> {
        let map_type = self.cursor.read_string()?;
        let (matrix, feature_name) = match map_type.as_str() {
            "UniformScaleMap" | "ScaleMap" => {
                let scale = self.cursor.read_dvec3()?;
                self.cursor.read_dvec3()?;
                self.cursor.read_dvec3()?;
                self.cursor.read_dvec3()?;
                self.cursor.read_dvec3()?;
                (DMat4::from_scale(scale), "linear scale map")
            }
            "UniformScaleTranslateMap" | "ScaleTranslateMap" => {
                let translation = self.cursor.read_dvec3()?;
                let scale = self.cursor.read_dvec3()?;
                self.cursor.read_dvec3()?;
                self.cursor.read_dvec3()?;
                self.cursor.read_dvec3()?;
                self.cursor.read_dvec3()?;
                (
                    DMat4::from_translation(translation) * DMat4::from_scale(scale),
                    "scale-translate map",
                )
            }
            "TranslationMap" => (
                DMat4::from_translation(self.cursor.read_dvec3()?),
                "translation map",
            ),
            "AffineMap" => {
                let mut row_major = [0.0; 16];
                for value in &mut row_major {
                    *value = self.cursor.read_f64()?;
                }
                let column_major = [
                    row_major[0],
                    row_major[4],
                    row_major[8],
                    row_major[12],
                    row_major[1],
                    row_major[5],
                    row_major[9],
                    row_major[13],
                    row_major[2],
                    row_major[6],
                    row_major[10],
                    row_major[14],
                    row_major[3],
                    row_major[7],
                    row_major[11],
                    row_major[15],
                ];
                (DMat4::from_cols_array(&column_major), "affine map")
            }
            _ => {
                return Err(PotError::with_details(
                    ErrorCode::UnsupportedFeature,
                    format!("OpenVDB transform `{map_type}` is not supported"),
                    json!({"feature_id":"volume.openvdb_evaluation.transform","transform":map_type}),
                ));
            }
        };
        let inverse = matrix.inverse();
        if !matrix.is_finite() || !inverse.is_finite() || matrix.determinant().abs() <= f64::EPSILON
        {
            return Err(PotError::with_details(
                ErrorCode::EvaluationFailed,
                format!("OpenVDB {feature_name} is singular or non-finite"),
                json!({"feature_id":"volume.openvdb_evaluation.transform"}),
            ));
        }
        Ok(VdbTransform {
            index_to_world: matrix,
            world_to_index: inverse,
            map_type: match map_type.as_str() {
                "UniformScaleMap" => "UniformScaleMap",
                "ScaleMap" => "ScaleMap",
                "UniformScaleTranslateMap" => "UniformScaleTranslateMap",
                "ScaleTranslateMap" => "ScaleTranslateMap",
                "TranslationMap" => "TranslationMap",
                _ => "AffineMap",
            },
        })
    }

    fn read_grid_tree(
        &mut self,
        descriptor: &Descriptor,
        metadata: GridMetadata,
        transform: &VdbTransform,
        value_type: ValueKind,
        topology_start: usize,
    ) -> Result<VdbGrid> {
        self.cursor.position = topology_start;
        let buffer_count = self.cursor.read_u32()?;
        if buffer_count != 1 {
            return Err(PotError::with_details(
                ErrorCode::UnsupportedFeature,
                "OpenVDB multi-buffer trees are not supported",
                json!({"feature_id":"volume.openvdb_evaluation.tree_buffers","count":buffer_count}),
            ));
        }
        let background = self.cursor.read_value(value_type, false)?;
        let tile_count = usize::try_from(self.cursor.read_u32()?)
            .map_err(|_| import_error("VDB root tile count is outside platform limits"))?;
        let child_count = usize::try_from(self.cursor.read_u32()?)
            .map_err(|_| import_error("VDB root child count is outside platform limits"))?;
        self.add_nodes(
            tile_count
                .checked_add(child_count)
                .ok_or_else(|| limit_error("VDB root node count overflow"))?,
        )?;
        let mut tree = VdbTree {
            background: background.magnitude(),
            background_value: background,
            values: HashMap::new(),
            tiles: HashMap::new(),
            active_bbox: None,
        };
        for _ in 0..tile_count {
            let origin = self.cursor.read_i32_vec3()?;
            let value = self.cursor.read_value(value_type, false)?;
            let active = self.cursor.read_u8()? != 0;
            self.insert_tile(&mut tree, origin, 4096, value, active)?;
        }
        let mut leaves = Vec::new();
        for _ in 0..child_count {
            let origin = self.cursor.read_i32_vec3()?;
            self.read_node(
                &mut tree,
                &mut leaves,
                origin,
                5,
                value_type,
                metadata.compression,
                metadata.half_float,
                background,
            )?;
        }
        if self.cursor.position > descriptor.block_pos {
            return Err(import_error("VDB tree topology overlaps its value blocks"));
        }
        self.cursor.position = descriptor.block_pos;
        for leaf in leaves {
            let block_mask = self.read_mask(512)?;
            if block_mask != leaf.active {
                return Err(import_error("VDB leaf topology and value masks differ"));
            }
            self.add_voxels(512)?;
            let values = self.read_compressed_values(
                512,
                &leaf.active,
                value_type,
                metadata.compression,
                metadata.half_float,
                background,
            )?;
            for (offset, value) in values.into_iter().enumerate() {
                let local = leaf_offset(offset);
                let coordinate = add_coord(leaf.origin, local)?;
                let active = leaf.active[offset];
                if active {
                    Self::include_active(&mut tree.active_bbox, coordinate, coordinate);
                }
                if active || value != background {
                    tree.values.insert(IVec3::from_array(coordinate), value);
                }
            }
        }
        if self.cursor.position > descriptor.end_pos {
            return Err(import_error(
                "VDB grid value blocks extend beyond the descriptor end",
            ));
        }
        let active_bbox = tree.active_bbox;
        Ok(VdbGrid {
            name: descriptor.name.clone(),
            class: metadata.class,
            value_type: descriptor.grid_type.clone(),
            transform: *transform,
            tree: Arc::new(tree),
            active_bbox,
        })
    }

    fn read_node(
        &mut self,
        tree: &mut VdbTree,
        leaves: &mut Vec<LeafDescriptor>,
        origin: [i32; 3],
        level: u8,
        value_type: ValueKind,
        compression: u32,
        half_float: bool,
        background: VdbValue,
    ) -> Result<()> {
        if level == 3 {
            self.add_nodes(1)?;
            leaves.push(LeafDescriptor {
                origin,
                active: self.read_mask(512)?,
            });
            return Ok(());
        }
        let log2_dim = usize::from(level);
        let count = 1_usize
            .checked_shl(
                u32::try_from(3 * log2_dim)
                    .map_err(|_| limit_error("VDB node dimension overflow"))?,
            )
            .ok_or_else(|| limit_error("VDB node dimension overflow"))?;
        let child_mask = self.read_mask(count)?;
        let value_mask = self.read_mask(count)?;
        let values = self.read_compressed_values(
            count,
            &value_mask,
            value_type,
            compression,
            half_float,
            background,
        )?;
        let child_size = if level == 5 { 128 } else { 8 };
        let tile_size = child_size;
        self.add_nodes(1)?;
        for (index, value) in values.iter().copied().enumerate() {
            if child_mask[index] {
                continue;
            }
            let tile_origin = offset_origin(origin, index, log2_dim, child_size)?;
            let active = value_mask[index];
            self.insert_tile(tree, tile_origin, tile_size, value, active)?;
        }
        for (index, has_child) in child_mask.iter().copied().enumerate() {
            if has_child {
                let child_origin = offset_origin(origin, index, log2_dim, child_size)?;
                self.read_node(
                    tree,
                    leaves,
                    child_origin,
                    level - 1,
                    value_type,
                    compression,
                    half_float,
                    background,
                )?;
            }
        }
        Ok(())
    }

    fn read_compressed_values(
        &mut self,
        count: usize,
        value_mask: &[bool],
        value_type: ValueKind,
        compression: u32,
        half_float: bool,
        background: VdbValue,
    ) -> Result<Vec<VdbValue>> {
        if value_mask.len() != count {
            return Err(import_error("VDB value mask size is inconsistent"));
        }
        let mode = self.cursor.read_u8()?;
        if mode > 6 {
            return Err(import_error(
                "VDB node has an invalid mask-compression mode",
            ));
        }
        let mut inactive0 = if mode == 0 {
            background
        } else {
            negate_value(background)
        };
        let mut inactive1 = background;
        if matches!(mode, 2 | 4 | 5) {
            inactive0 = self.cursor.read_value(value_type, false)?;
            if mode == 5 {
                inactive1 = self.cursor.read_value(value_type, false)?;
            }
        }
        let selection_mask = if matches!(mode, 3..=5) {
            self.read_mask(count)?
        } else {
            vec![false; count]
        };
        let mask_compressed = compression & ACTIVE_MASK_COMPRESSION != 0 && mode != 6;
        let value_count = if mask_compressed {
            value_mask.iter().filter(|active| **active).count()
        } else {
            count
        };
        let compressed =
            self.read_value_data(value_count, value_type, compression, half_float, mode)?;
        if compressed.len() != value_count {
            return Err(import_error("VDB compressed value count is inconsistent"));
        }
        if !mask_compressed || value_count == count {
            return Ok(compressed);
        }
        let mut result = Vec::with_capacity(count);
        let mut source_index = 0;
        for index in 0..count {
            if value_mask[index] {
                result.push(
                    *compressed
                        .get(source_index)
                        .ok_or_else(|| import_error("VDB active value data is truncated"))?,
                );
                source_index += 1;
            } else if selection_mask[index] {
                result.push(inactive1);
            } else {
                result.push(inactive0);
            }
        }
        Ok(result)
    }

    fn read_value_data(
        &mut self,
        count: usize,
        value_type: ValueKind,
        compression: u32,
        half_float: bool,
        mode: u8,
    ) -> Result<Vec<VdbValue>> {
        let Some(bytes_per_value) = value_type.byte_size(half_float) else {
            return Err(import_error("unsupported VDB value representation"));
        };
        let expected = count
            .checked_mul(bytes_per_value)
            .ok_or_else(|| limit_error("VDB decoded value byte count overflow"))?;
        if expected > MAX_FILE_BYTES {
            return Err(limit_error(
                "VDB decoded value data exceeds the memory budget",
            ));
        }
        if count == 0 && (half_float || compression & (BLOSC_COMPRESSION | ZIP_COMPRESSION) == 0) {
            return Ok(Vec::new());
        }
        let bytes = if compression & (BLOSC_COMPRESSION | ZIP_COMPRESSION) == 0 {
            self.cursor.take(expected)?.to_vec()
        } else {
            let length_position = self.cursor.position;
            let length = self.cursor.read_i64()?;
            let absolute = length.unsigned_abs();
            let compressed_size = usize::try_from(absolute)
                .map_err(|_| limit_error("VDB compressed block size is outside platform limits"))?;
            if compressed_size > MAX_COMPRESSED_BYTES {
                return Err(limit_error(format!(
                    "VDB compressed block at {length_position} has size {compressed_size}, over budget"
                )));
            }
            let source = self.cursor.take(compressed_size)?;
            if length <= 0 {
                if compressed_size != expected {
                    return Err(import_error(format!(
                        "VDB raw block at {length_position} has {compressed_size} bytes; expected {expected} for {count} values (mode {mode}, compression {compression:#x})"
                    )));
                }
                source.to_vec()
            } else if compression & BLOSC_COMPRESSION != 0 {
                decompress_blosc(source, expected)?
            } else {
                let decoder = flate2::read::ZlibDecoder::new(source);
                let mut output = Vec::with_capacity(expected);
                decoder
                    .take(
                        u64::try_from(expected)
                            .unwrap_or(u64::MAX)
                            .saturating_add(1),
                    )
                    .read_to_end(&mut output)
                    .map_err(|_| import_error("VDB zlib block could not be decompressed"))?;
                if output.len() != expected {
                    return Err(import_error("VDB zlib block has an invalid decoded size"));
                }
                output
            }
        };
        if bytes.len() != expected {
            return Err(import_error(
                "VDB decompressed value byte count is inconsistent",
            ));
        }
        let mut result = Vec::with_capacity(count);
        for value in bytes.chunks_exact(bytes_per_value) {
            result.push(decode_value(value, value_type, half_float)?);
        }
        Ok(result)
    }

    fn read_mask(&mut self, count: usize) -> Result<Vec<bool>> {
        let word_count = count.div_ceil(64);
        let mut mask = Vec::with_capacity(count);
        for _ in 0..word_count {
            let word = self.cursor.read_u64()?;
            for bit in 0..64 {
                if mask.len() == count {
                    break;
                }
                mask.push((word >> bit) & 1 != 0);
            }
        }
        Ok(mask)
    }

    #[expect(
        clippy::float_cmp,
        reason = "OpenVDB background elision uses exact serialized value equality"
    )]
    fn insert_tile(
        &mut self,
        tree: &mut VdbTree,
        origin: [i32; 3],
        width: i32,
        value: VdbValue,
        active: bool,
    ) -> Result<()> {
        let max = [
            origin[0].checked_add(width - 1),
            origin[1].checked_add(width - 1),
            origin[2].checked_add(width - 1),
        ];
        let max = [
            max[0].ok_or_else(|| import_error("VDB tile coordinate overflow"))?,
            max[1].ok_or_else(|| import_error("VDB tile coordinate overflow"))?,
            max[2].ok_or_else(|| import_error("VDB tile coordinate overflow"))?,
        ];
        if active {
            let voxel_count = u64::try_from(width)
                .ok()
                .and_then(|value| value.checked_mul(value))
                .and_then(|value| value.checked_mul(u64::try_from(width).ok()?))
                .ok_or_else(|| limit_error("VDB active tile voxel count overflow"))?;
            self.total_voxels = self
                .total_voxels
                .checked_add(usize::try_from(voxel_count).unwrap_or(usize::MAX))
                .ok_or_else(|| limit_error("VDB total voxel count overflow"))?;
            if self.total_voxels > MAX_VOXELS {
                return Err(limit_error(
                    "VDB active voxel count exceeds the volume budget",
                ));
            }
            Self::include_active(&mut tree.active_bbox, origin, max);
        }
        if active || value.magnitude() != tree.background {
            self.total_tiles = self
                .total_tiles
                .checked_add(1)
                .ok_or_else(|| limit_error("VDB tile count overflow"))?;
            if self.total_tiles > MAX_VOXELS {
                return Err(limit_error("VDB tile count exceeds the memory budget"));
            }
            if tree
                .tiles
                .insert(
                    (IVec3::from_array(origin), width),
                    VdbTile {
                        min: origin,
                        max,
                        value,
                        active,
                    },
                )
                .is_some()
            {
                return Err(import_error("VDB tree contains duplicate tile entries"));
            }
        }
        Ok(())
    }

    fn include_active(
        bounds: &mut Option<([i32; 3], [i32; 3])>,
        minimum: [i32; 3],
        maximum: [i32; 3],
    ) {
        *bounds = Some(match *bounds {
            Some((old_min, old_max)) => (
                [
                    old_min[0].min(minimum[0]),
                    old_min[1].min(minimum[1]),
                    old_min[2].min(minimum[2]),
                ],
                [
                    old_max[0].max(maximum[0]),
                    old_max[1].max(maximum[1]),
                    old_max[2].max(maximum[2]),
                ],
            ),
            None => (minimum, maximum),
        });
    }

    fn add_nodes(&mut self, count: usize) -> Result<()> {
        self.total_nodes = self
            .total_nodes
            .checked_add(count)
            .ok_or_else(|| limit_error("VDB node count overflow"))?;
        if self.total_nodes > MAX_NODE_COUNT {
            return Err(limit_error("VDB tree contains too many nodes"));
        }
        Ok(())
    }

    fn add_voxels(&mut self, count: usize) -> Result<()> {
        self.total_voxels = self
            .total_voxels
            .checked_add(count)
            .ok_or_else(|| limit_error("VDB leaf voxel count overflow"))?;
        if self.total_voxels > MAX_VOXELS {
            return Err(limit_error(
                "VDB leaf voxel count exceeds the volume budget",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ValueKind {
    Float,
    Vec3,
}

impl ValueKind {
    fn byte_size(self, half: bool) -> Option<usize> {
        let bytes_per_component = if half { 2 } else { 4 };
        match self {
            Self::Float => Some(bytes_per_component),
            Self::Vec3 => bytes_per_component.checked_mul(3),
        }
    }
}

#[derive(Clone, Debug)]
enum MetadataValue {
    String(String),
    Bool(bool),
}

fn grid_value_type(grid_type: &str) -> Result<ValueKind> {
    let parts = grid_type.split('_').collect::<Vec<_>>();
    if parts.first() != Some(&"Tree") || parts.get(2..5) != Some(["5", "4", "3"].as_slice()) {
        return Err(PotError::with_details(
            ErrorCode::UnsupportedFeature,
            format!("OpenVDB tree configuration `{grid_type}` is not supported"),
            json!({"feature_id":"volume.openvdb_evaluation.tree_buffers","grid_type":grid_type}),
        ));
    }
    match parts.get(1).copied() {
        Some("float") => Ok(ValueKind::Float),
        Some("vec3s") => Ok(ValueKind::Vec3),
        _ => Err(PotError::with_details(
            ErrorCode::UnsupportedFeature,
            format!("OpenVDB grid type `{grid_type}` is not supported"),
            json!({"feature_id":"volume.openvdb_evaluation.value_type","grid_type":grid_type}),
        )),
    }
}

fn validate_compression(flags: u32) -> Result<()> {
    let supported = ZIP_COMPRESSION | ACTIVE_MASK_COMPRESSION | BLOSC_COMPRESSION;
    if flags & !supported != 0
        || flags & (ZIP_COMPRESSION | BLOSC_COMPRESSION) == (ZIP_COMPRESSION | BLOSC_COMPRESSION)
    {
        return Err(PotError::with_details(
            ErrorCode::UnsupportedFeature,
            format!("OpenVDB compression flags {flags:#x} are not supported"),
            json!({"feature_id":"volume.openvdb_evaluation.compression","compression":flags}),
        ));
    }
    Ok(())
}

fn decompress_blosc(source: &[u8], expected: usize) -> Result<Vec<u8>> {
    if source.len() < BLOSC_HEADER_SIZE {
        return Err(import_error("VDB Blosc header is truncated"));
    }
    let version = source[0];
    let versionlz = source[1];
    let flags = source[2];
    let typesize = usize::from(source[3]);
    let nbytes = usize::try_from(read_u32_at(source, 4)?)
        .map_err(|_| limit_error("VDB Blosc decoded size is outside platform limits"))?;
    let blocksize = usize::try_from(read_u32_at(source, 8)?)
        .map_err(|_| limit_error("VDB Blosc block size is outside platform limits"))?;
    let cbytes = usize::try_from(read_u32_at(source, 12)?)
        .map_err(|_| limit_error("VDB Blosc compressed size is outside platform limits"))?;

    if !matches!(version, 1 | 2) || versionlz != 1 {
        return Err(unsupported_blosc(
            "version",
            format!("Blosc format version {version}, codec version {versionlz}"),
        ));
    }
    if flags & 0x08 != 0 {
        return Err(unsupported_blosc(
            "flags",
            format!("reserved flag is set in {flags:#04x}"),
        ));
    }
    if flags & BLOSC_DOBITSHUFFLE != 0 {
        return Err(unsupported_blosc(
            "bitshuffle",
            "bit-shuffled Blosc blocks are not supported".to_owned(),
        ));
    }
    let codec = (flags >> 5) & 0x07;
    if !matches!(codec, 1 | 3 | 4) {
        let name = match codec {
            0 => "BloscLZ",
            2 => "Snappy",
            _ => "unknown",
        };
        return Err(unsupported_blosc(
            "codec",
            format!("Blosc codec `{name}` ({codec}) is not supported"),
        ));
    }
    if cbytes != source.len() || nbytes != expected {
        return Err(import_error(format!(
            "VDB Blosc header declares {nbytes} decoded and {cbytes} compressed bytes; expected {expected} decoded bytes and {} compressed bytes",
            source.len()
        )));
    }
    if typesize == 0 {
        return Err(import_error("VDB Blosc typesize is zero"));
    }

    if flags & BLOSC_MEMCPYED != 0 {
        let expected_cbytes = BLOSC_HEADER_SIZE
            .checked_add(expected)
            .ok_or_else(|| limit_error("VDB Blosc memcpy size overflow"))?;
        if cbytes != expected_cbytes {
            return Err(import_error("VDB Blosc memcpy block has an invalid size"));
        }
        return Ok(source[BLOSC_HEADER_SIZE..].to_vec());
    }
    if expected == 0 {
        if blocksize != 0 {
            return Err(import_error(
                "empty VDB Blosc block has a nonzero block size",
            ));
        }
        return Ok(Vec::new());
    }
    if blocksize == 0 || blocksize > MAX_FILE_BYTES || typesize > blocksize {
        return Err(import_error("VDB Blosc block size or typesize is invalid"));
    }

    let block_count = expected.div_ceil(blocksize);
    let table_bytes = block_count
        .checked_mul(std::mem::size_of::<u32>())
        .ok_or_else(|| limit_error("VDB Blosc block-start table size overflow"))?;
    let payload_start = BLOSC_HEADER_SIZE
        .checked_add(table_bytes)
        .ok_or_else(|| limit_error("VDB Blosc block-start table overflow"))?;
    if payload_start > source.len() {
        return Err(import_error("VDB Blosc block-start table is truncated"));
    }

    let shuffle = flags & BLOSC_DOSHUFFLE != 0;
    let mut decoded = vec![0_u8; expected];
    let mut unshuffle_buffer = if shuffle {
        vec![0_u8; blocksize.min(expected)]
    } else {
        Vec::new()
    };
    let mut previous_start = payload_start;
    for block_index in 0..block_count {
        let start_position = BLOSC_HEADER_SIZE
            .checked_add(block_index * std::mem::size_of::<u32>())
            .ok_or_else(|| limit_error("VDB Blosc block-start offset overflow"))?;
        let start = usize::try_from(read_u32_at(source, start_position)?)
            .map_err(|_| limit_error("VDB Blosc block offset is outside platform limits"))?;
        let end = if block_index + 1 == block_count {
            cbytes
        } else {
            let next_position = start_position + std::mem::size_of::<u32>();
            usize::try_from(read_u32_at(source, next_position)?)
                .map_err(|_| limit_error("VDB Blosc block offset is outside platform limits"))?
        };
        if start < payload_start || start < previous_start || start >= end || end > cbytes {
            return Err(import_error(
                "VDB Blosc block-start table contains an invalid offset",
            ));
        }
        previous_start = start;

        let output_start = block_index
            .checked_mul(blocksize)
            .ok_or_else(|| limit_error("VDB Blosc output offset overflow"))?;
        let block_output_len = (expected - output_start).min(blocksize);
        let block = source
            .get(start..end)
            .ok_or_else(|| import_error("VDB Blosc block range is invalid"))?;
        let block_output = decoded
            .get_mut(output_start..output_start + block_output_len)
            .ok_or_else(|| import_error("VDB Blosc output range is invalid"))?;
        let split_count = if flags & BLOSC_DONT_SPLIT == 0
            && block_output_len == blocksize
            && typesize <= BLOSC_MAX_SPLITS
            && blocksize / typesize >= 128
        {
            typesize
        } else {
            1
        };
        if block_output_len % split_count != 0 {
            return Err(import_error("VDB Blosc split does not divide its block"));
        }
        let split_size = block_output_len / split_count;
        let mut input_offset = 0_usize;
        for split_index in 0..split_count {
            let size_prefix_end = input_offset
                .checked_add(std::mem::size_of::<u32>())
                .ok_or_else(|| limit_error("VDB Blosc split header offset overflow"))?;
            let compressed_len = usize::try_from(read_u32_at(block, input_offset)?)
                .map_err(|_| limit_error("VDB Blosc split size is outside platform limits"))?;
            let compressed_end = size_prefix_end
                .checked_add(compressed_len)
                .ok_or_else(|| limit_error("VDB Blosc split range overflow"))?;
            let compressed = block
                .get(size_prefix_end..compressed_end)
                .ok_or_else(|| import_error("VDB Blosc split data is truncated"))?;
            let output_offset = split_index
                .checked_mul(split_size)
                .ok_or_else(|| limit_error("VDB Blosc split output offset overflow"))?;
            let split_output = block_output
                .get_mut(output_offset..output_offset + split_size)
                .ok_or_else(|| import_error("VDB Blosc split output range is invalid"))?;
            if compressed_len == split_size {
                split_output.copy_from_slice(compressed);
            } else {
                decompress_blosc_stream(codec, compressed, split_output)?;
            }
            input_offset = compressed_end;
        }
        if input_offset != block.len() {
            return Err(import_error("VDB Blosc block contains trailing split data"));
        }
        if shuffle {
            unshuffle_blosc_block(block_output, typesize, &mut unshuffle_buffer)?;
        }
    }
    Ok(decoded)
}

fn decompress_blosc_stream(codec: u8, source: &[u8], output: &mut [u8]) -> Result<()> {
    match codec {
        1 => {
            let decoded = lz4_flex::block::decompress_into(source, output)
                .map_err(|_| import_error("VDB Blosc LZ4 block could not be decompressed"))?;
            if decoded != output.len() {
                return Err(import_error(
                    "VDB Blosc LZ4 block has an invalid decoded size",
                ));
            }
        }
        3 => {
            let mut decoder = flate2::read::ZlibDecoder::new(source);
            decoder
                .read_exact(output)
                .map_err(|_| import_error("VDB Blosc zlib block could not be decompressed"))?;
            let mut extra = [0_u8; 1];
            if decoder
                .read(&mut extra)
                .map_err(|_| import_error("VDB Blosc zlib block could not be decompressed"))?
                != 0
            {
                return Err(import_error(
                    "VDB Blosc zlib block has an invalid decoded size",
                ));
            }
        }
        4 => {
            let maximum_window = u64::try_from(output.len())
                .map_err(|_| limit_error("VDB Blosc zstd window is outside platform limits"))?
                .max(BLOSC_MIN_ZSTD_WINDOW_BYTES);
            let mut decoder = ruzstd::decoding::StreamingDecoder::new_with_max_window_size(
                source,
                maximum_window,
            )
            .map_err(|_| import_error("VDB Blosc zstd block could not be decompressed"))?;
            decoder
                .read_exact(output)
                .map_err(|_| import_error("VDB Blosc zstd block could not be decompressed"))?;
            let mut extra = [0_u8; 1];
            if decoder
                .read(&mut extra)
                .map_err(|_| import_error("VDB Blosc zstd block could not be decompressed"))?
                != 0
            {
                return Err(import_error(
                    "VDB Blosc zstd block has an invalid decoded size",
                ));
            }
        }
        _ => {
            return Err(unsupported_blosc(
                "codec",
                format!("Blosc codec {codec} is not supported"),
            ));
        }
    }
    Ok(())
}

fn unshuffle_blosc_block(block: &mut [u8], typesize: usize, scratch: &mut [u8]) -> Result<()> {
    if typesize <= 1 {
        return Ok(());
    }
    if scratch.len() < block.len() {
        return Err(import_error("VDB Blosc unshuffle buffer is too small"));
    }
    let value_count = block.len() / typesize;
    let shuffled_len = value_count * typesize;
    for component in 0..typesize {
        for value_index in 0..value_count {
            scratch[value_index * typesize + component] =
                block[component * value_count + value_index];
        }
    }
    scratch[shuffled_len..block.len()].copy_from_slice(&block[shuffled_len..]);
    block.copy_from_slice(&scratch[..block.len()]);
    Ok(())
}

fn read_u32_at(bytes: &[u8], offset: usize) -> Result<u32> {
    let end = offset
        .checked_add(std::mem::size_of::<u32>())
        .ok_or_else(|| limit_error("VDB Blosc integer offset overflow"))?;
    let bytes = bytes
        .get(offset..end)
        .ok_or_else(|| import_error("VDB Blosc integer is truncated"))?;
    Ok(u32::from_le_bytes(bytes.try_into().map_err(|_| {
        import_error("VDB Blosc integer is truncated")
    })?))
}

fn unsupported_blosc(feature: &str, message: String) -> PotError {
    PotError::with_details(
        ErrorCode::UnsupportedFeature,
        message,
        json!({"feature_id":format!("volume.openvdb_evaluation.blosc_{feature}")}),
    )
}

fn decode_value(bytes: &[u8], kind: ValueKind, half: bool) -> Result<VdbValue> {
    let component = |offset: usize| -> Result<f32> {
        if half {
            let bits = u16::from_le_bytes(
                bytes
                    .get(offset..offset + 2)
                    .ok_or_else(|| import_error("VDB half value is truncated"))?
                    .try_into()
                    .map_err(|_| import_error("VDB half value is truncated"))?,
            );
            Ok(f16_to_f32(bits))
        } else {
            let bits: [u8; 4] = bytes
                .get(offset..offset + 4)
                .ok_or_else(|| import_error("VDB float value is truncated"))?
                .try_into()
                .map_err(|_| import_error("VDB float value is truncated"))?;
            Ok(f32::from_le_bytes(bits))
        }
    };
    match kind {
        ValueKind::Float => Ok(VdbValue::Float(component(0)?)),
        ValueKind::Vec3 => {
            let size = if half { 2 } else { 4 };
            Ok(VdbValue::Vec3([
                component(0)?,
                component(size)?,
                component(size * 2)?,
            ]))
        }
    }
}

fn f16_to_f32(bits: u16) -> f32 {
    let sign = u32::from(bits & 0x8000) << 16;
    let exponent = u32::from((bits >> 10) & 0x1f);
    let mantissa = u32::from(bits & 0x03ff);
    let converted = match exponent {
        0 if mantissa == 0 => sign,
        0 => {
            let mut normalized = mantissa;
            let mut shift = 0_u32;
            while normalized & 0x0400 == 0 {
                normalized <<= 1;
                shift += 1;
            }
            sign | ((127 - 14 - shift) << 23) | ((normalized & 0x03ff) << 13)
        }
        0x1f => sign | 0x7f80_0000 | (mantissa << 13),
        _ => sign | ((exponent + 127 - 15) << 23) | (mantissa << 13),
    };
    f32::from_bits(converted)
}

fn negate_value(value: VdbValue) -> VdbValue {
    match value {
        VdbValue::Float(value) => VdbValue::Float(-value),
        VdbValue::Vec3(value) => VdbValue::Vec3(value.map(|component| -component)),
    }
}

fn leaf_offset(index: usize) -> [i32; 3] {
    [
        i32::try_from(index >> 6).unwrap_or(0),
        i32::try_from((index >> 3) & 7).unwrap_or(0),
        i32::try_from(index & 7).unwrap_or(0),
    ]
}

fn offset_origin(
    origin: [i32; 3],
    index: usize,
    log2_dim: usize,
    child_size: i32,
) -> Result<[i32; 3]> {
    let mask = (1_usize << log2_dim) - 1;
    let local = [
        index >> (2 * log2_dim),
        (index >> log2_dim) & mask,
        index & mask,
    ];
    let mut result = origin;
    for axis in 0..3 {
        let offset = i32::try_from(local[axis])
            .ok()
            .and_then(|value| value.checked_mul(child_size))
            .ok_or_else(|| import_error("VDB child coordinate overflow"))?;
        result[axis] = result[axis]
            .checked_add(offset)
            .ok_or_else(|| import_error("VDB child coordinate overflow"))?;
    }
    Ok(result)
}

fn add_coord(origin: [i32; 3], offset: [i32; 3]) -> Result<[i32; 3]> {
    let mut result = origin;
    for axis in 0..3 {
        result[axis] = result[axis]
            .checked_add(offset[axis])
            .ok_or_else(|| import_error("VDB voxel coordinate overflow"))?;
    }
    Ok(result)
}

fn import_error(message: impl Into<String>) -> PotError {
    PotError::with_details(
        ErrorCode::ImportFailed,
        message,
        json!({"feature_id":"volume.openvdb_evaluation"}),
    )
}

fn limit_error(message: impl Into<String>) -> PotError {
    PotError::with_details(
        ErrorCode::LimitExceeded,
        message,
        json!({"feature_id":"volume.openvdb_evaluation.resource_budget"}),
    )
}

struct Cursor<'a> {
    bytes: &'a [u8],
    position: usize,
}

impl<'a> Cursor<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, position: 0 }
    }

    fn take(&mut self, count: usize) -> Result<&'a [u8]> {
        let end = self
            .position
            .checked_add(count)
            .ok_or_else(|| import_error("VDB byte offset overflow"))?;
        let value = self
            .bytes
            .get(self.position..end)
            .ok_or_else(|| import_error("VDB archive is truncated"))?;
        self.position = end;
        Ok(value)
    }

    fn skip(&mut self, count: usize) -> Result<()> {
        self.take(count).map(|_| ())
    }

    fn read_u8(&mut self) -> Result<u8> {
        self.take(1)?
            .first()
            .copied()
            .ok_or_else(|| import_error("VDB archive is truncated"))
    }

    fn read_u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(
            self.take(4)?
                .try_into()
                .map_err(|_| import_error("VDB u32 is truncated"))?,
        ))
    }

    fn read_i32(&mut self) -> Result<i32> {
        Ok(i32::from_le_bytes(
            self.take(4)?
                .try_into()
                .map_err(|_| import_error("VDB i32 is truncated"))?,
        ))
    }

    fn read_u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(
            self.take(8)?
                .try_into()
                .map_err(|_| import_error("VDB u64 is truncated"))?,
        ))
    }

    fn read_i64(&mut self) -> Result<i64> {
        Ok(i64::from_le_bytes(
            self.take(8)?
                .try_into()
                .map_err(|_| import_error("VDB i64 is truncated"))?,
        ))
    }

    fn read_f64(&mut self) -> Result<f64> {
        Ok(f64::from_le_bytes(
            self.take(8)?
                .try_into()
                .map_err(|_| import_error("VDB f64 is truncated"))?,
        ))
    }

    fn read_offset(&mut self) -> Result<usize> {
        usize::try_from(self.read_u64()?)
            .map_err(|_| import_error("VDB offset is outside platform limits"))
    }

    fn read_string(&mut self) -> Result<String> {
        let length = usize::try_from(self.read_u32()?)
            .map_err(|_| import_error("VDB string length is outside platform limits"))?;
        if length > MAX_STRING_BYTES {
            return Err(limit_error(format!(
                "VDB string at {} has length {length}, over budget",
                self.position.saturating_sub(std::mem::size_of::<u32>())
            )));
        }
        String::from_utf8(self.take(length)?.to_vec())
            .map_err(|_| import_error("VDB string is not UTF-8"))
    }

    fn read_string_fixed(&mut self, length: usize) -> Result<String> {
        String::from_utf8(self.take(length)?.to_vec())
            .map_err(|_| import_error("VDB UUID is not UTF-8"))
    }

    fn read_dvec3(&mut self) -> Result<DVec3> {
        Ok(DVec3::new(
            self.read_f64()?,
            self.read_f64()?,
            self.read_f64()?,
        ))
    }

    fn read_i32_vec3(&mut self) -> Result<[i32; 3]> {
        Ok([self.read_i32()?, self.read_i32()?, self.read_i32()?])
    }

    fn read_value(&mut self, kind: ValueKind, half: bool) -> Result<VdbValue> {
        let length = kind
            .byte_size(half)
            .ok_or_else(|| import_error("VDB value byte size overflow"))?;
        decode_value(self.take(length)?, kind, half)
    }
}

#[cfg(kani)]
#[kani::proof]
fn offset_range_addition_is_checked() {
    let mut cursor = Cursor::new(&[]);
    cursor.position = usize::MAX;
    assert!(cursor.take(1).is_err());
}

#[cfg(test)]
mod blosc_tests {
    #![expect(clippy::unwrap_used, reason = "small deterministic codec fixtures")]

    use std::io::Write;

    use flate2::write::ZlibEncoder;

    use super::*;

    const TEST_TYPESIZE: usize = 4;
    const TEST_BLOCKSIZE: usize = 512;

    #[test]
    fn decodes_lz4_lz4hc_zlib_and_zstd_split_layouts_with_shuffle() {
        let original = (0..TEST_BLOCKSIZE * 2)
            .flat_map(|index| [u8::try_from(index % 127).unwrap(), 0, 0, 0])
            .collect::<Vec<_>>();
        for codec in [1_u8, 3, 4] {
            for dont_split in [false, true] {
                let frame = make_blosc_frame(codec, &original, true, dont_split);
                assert_eq!(
                    decompress_blosc(&frame, original.len()).unwrap(),
                    original,
                    "codec {codec}, dont_split={dont_split} did not round-trip"
                );
            }
        }
    }

    #[test]
    fn copies_memcpyed_frames_without_a_block_table() {
        let original = (0..TEST_BLOCKSIZE)
            .map(|value| u8::try_from(value % 251).unwrap())
            .collect::<Vec<_>>();
        let frame = make_memcpy_frame(&original, TEST_TYPESIZE, 1);
        assert_eq!(decompress_blosc(&frame, original.len()).unwrap(), original);
    }

    #[test]
    fn reports_typed_errors_for_blosc_variants_not_supported_by_blender() {
        let bytes = vec![0_u8; 64];
        let blosclz = make_memcpy_frame(&bytes, 1, 0);
        let error = decompress_blosc(&blosclz, bytes.len()).unwrap_err();
        assert_eq!(error.code, ErrorCode::UnsupportedFeature);
        assert_eq!(
            error.details["feature_id"],
            "volume.openvdb_evaluation.blosc_codec"
        );

        let mut bitshuffle = make_memcpy_frame(&bytes, 1, 1);
        bitshuffle[2] |= BLOSC_DOBITSHUFFLE;
        let error = decompress_blosc(&bitshuffle, bytes.len()).unwrap_err();
        assert_eq!(error.code, ErrorCode::UnsupportedFeature);
        assert_eq!(
            error.details["feature_id"],
            "volume.openvdb_evaluation.blosc_bitshuffle"
        );
    }

    fn make_blosc_frame(codec: u8, original: &[u8], shuffle: bool, dont_split: bool) -> Vec<u8> {
        assert_eq!(original.len() % TEST_BLOCKSIZE, 0);
        let mut transformed = Vec::with_capacity(original.len());
        for block in original.as_chunks::<TEST_BLOCKSIZE>().0 {
            if shuffle {
                transformed.extend_from_slice(&shuffle_bytes(block, TEST_TYPESIZE));
            } else {
                transformed.extend_from_slice(block);
            }
        }
        let split_count = if dont_split { 1 } else { TEST_TYPESIZE };
        let split_size = TEST_BLOCKSIZE / split_count;
        let mut body = Vec::new();
        let mut block_starts = Vec::new();
        for block in transformed.as_chunks::<TEST_BLOCKSIZE>().0 {
            block_starts
                .push(BLOSC_HEADER_SIZE + original.len().div_ceil(TEST_BLOCKSIZE) * 4 + body.len());
            for split in block.chunks_exact(split_size) {
                let compressed = compress_test_stream(codec, split);
                body.extend_from_slice(&u32::try_from(compressed.len()).unwrap().to_le_bytes());
                body.extend_from_slice(&compressed);
            }
        }
        let mut flags = codec << 5;
        if shuffle {
            flags |= BLOSC_DOSHUFFLE;
        }
        if dont_split {
            flags |= BLOSC_DONT_SPLIT;
        }
        let mut frame = Vec::with_capacity(
            BLOSC_HEADER_SIZE + original.len() / TEST_BLOCKSIZE * 4 + body.len(),
        );
        frame.extend_from_slice(&[2, 1, flags, u8::try_from(TEST_TYPESIZE).unwrap()]);
        frame.extend_from_slice(&u32::try_from(original.len()).unwrap().to_le_bytes());
        frame.extend_from_slice(&u32::try_from(TEST_BLOCKSIZE).unwrap().to_le_bytes());
        frame.extend_from_slice(&[0; 4]);
        for start in block_starts {
            frame.extend_from_slice(&u32::try_from(start).unwrap().to_le_bytes());
        }
        frame.extend_from_slice(&body);
        let compressed_len = u32::try_from(frame.len()).unwrap().to_le_bytes();
        frame[12..16].copy_from_slice(&compressed_len);
        frame
    }

    fn make_memcpy_frame(original: &[u8], typesize: usize, codec: u8) -> Vec<u8> {
        let mut frame = Vec::with_capacity(BLOSC_HEADER_SIZE + original.len());
        frame.extend_from_slice(&[
            2,
            1,
            (codec << 5) | BLOSC_MEMCPYED,
            u8::try_from(typesize).unwrap(),
        ]);
        frame.extend_from_slice(&u32::try_from(original.len()).unwrap().to_le_bytes());
        frame.extend_from_slice(&u32::try_from(original.len()).unwrap().to_le_bytes());
        frame.extend_from_slice(
            &u32::try_from(BLOSC_HEADER_SIZE + original.len())
                .unwrap()
                .to_le_bytes(),
        );
        frame.extend_from_slice(original);
        frame
    }

    fn compress_test_stream(codec: u8, source: &[u8]) -> Vec<u8> {
        match codec {
            1 => lz4_flex::block::compress(source),
            3 => {
                let mut encoder = ZlibEncoder::new(Vec::new(), flate2::Compression::default());
                encoder.write_all(source).unwrap();
                encoder.finish().unwrap()
            }
            4 => ruzstd::encoding::compress_to_vec(
                source,
                ruzstd::encoding::CompressionLevel::Fastest,
            ),
            _ => unreachable!(),
        }
    }

    fn shuffle_bytes(source: &[u8], typesize: usize) -> Vec<u8> {
        let value_count = source.len() / typesize;
        let mut result = vec![0_u8; source.len()];
        for component in 0..typesize {
            for value_index in 0..value_count {
                result[component * value_count + value_index] =
                    source[value_index * typesize + component];
            }
        }
        result
    }
}
#[cfg(test)]
mod sampling_tests {
    use super::*;

    #[test]
    fn world_sampling_uses_sparse_voxel_background_outside_the_sphere() {
        let mut values = HashMap::new();
        for z in 0..16 {
            for y in 0..16 {
                for x in 0..16 {
                    let offset = [f64::from(x) - 7.5, f64::from(y) - 7.5, f64::from(z) - 7.5];
                    if offset.into_iter().map(|value| value * value).sum::<f64>() < 36.0 {
                        values.insert(IVec3::new(x, y, z), VdbValue::Float(1.0));
                    }
                }
            }
        }
        let tree = Arc::new(VdbTree {
            background: 0.0,
            background_value: VdbValue::Float(0.0),
            values,
            tiles: HashMap::new(),
            active_bbox: Some(([2, 2, 2], [13, 13, 13])),
        });
        let grid = VdbGrid {
            name: "density".to_owned(),
            class: None,
            value_type: "Tree_float_5_4_3".to_owned(),
            transform: VdbTransform {
                index_to_world: DMat4::IDENTITY,
                world_to_index: DMat4::IDENTITY,
                map_type: "UnitMap",
            },
            tree,
            active_bbox: Some(([2, 2, 2], [13, 13, 13])),
        };

        assert_eq!(grid.sample_world(DVec3::new(2.0, 2.0, 7.5)), 0.0);
        assert_eq!(grid.sample_world(DVec3::splat(7.5)), 1.0);
    }
}
