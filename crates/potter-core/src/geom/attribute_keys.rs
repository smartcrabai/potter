//! Injective mapping between user-facing attribute names and the keys stored in
//! `Mesh::attributes`.
//!
//! Potter keeps its own stores (`custom_normal`, `color_layers`, `vertex_groups`, ...) in the
//! same map as user generic attributes. A user attribute whose name equals one of those keys, or
//! that begins with [`ESCAPE_PREFIX`], is therefore stored under the escaped key
//! `ESCAPE_PREFIX + name`. Every other name is stored as itself. The map is injective because
//! [`user_name`] inverts [`storage_key`] for every name.
//!
//! Layer names, active/render designations, material selectors and loss reasons always use the
//! user-facing name. Only storage keys are escaped.

use std::borrow::Cow;

/// Prefix of an escaped storage key. Names that already begin with it are escaped as well.
pub(crate) const ESCAPE_PREFIX: &str = "user:";

/// Keys Potter writes into `Mesh::attributes` for its own stores. Never a user attribute key.
const INTERNAL_KEYS: &[&str] = &[
    "alembic_normals",
    "alembic_time_samples",
    "alembic_velocity",
    "blender_attributes",
    "blender_metadata",
    "blender_skin_vertices",
    "color_layers",
    "crease_edge",
    "custom_normal",
    "multires_displacement",
    "seam",
    "sharp",
    "shade_smooth",
    "skin_radii",
    "skin_roots",
    "uv_layers",
    "vertex_groups",
];

/// Whether `key` is one of Potter's own stores rather than a user attribute.
pub(crate) fn is_internal_key(key: &str) -> bool {
    INTERNAL_KEYS.contains(&key)
}

/// Storage key for the user-facing attribute `name`.
pub(crate) fn storage_key(name: &str) -> Cow<'_, str> {
    if is_internal_key(name) || name.starts_with(ESCAPE_PREFIX) {
        Cow::Owned(format!("{ESCAPE_PREFIX}{name}"))
    } else {
        Cow::Borrowed(name)
    }
}

/// User-facing name stored under `key`, or `None` when `key` is one of Potter's own stores.
pub(crate) fn user_name(key: &str) -> Option<&str> {
    if let Some(name) = key.strip_prefix(ESCAPE_PREFIX) {
        Some(name)
    } else if is_internal_key(key) {
        None
    } else {
        Some(key)
    }
}

#[cfg(test)]
mod tests {
    use super::{ESCAPE_PREFIX, storage_key, user_name};

    #[test]
    fn reserved_and_escaped_names_round_trip_and_stay_distinct() {
        for name in [
            "color_layers",
            "vertex_groups",
            "sharp",
            "custom_normal",
            "skin_radii",
            "user:sharp",
            "user:user:color_layers",
            "blender_attributes",
            "plain",
            "blender_polygon_smooth",
            "skin_tone",
            "alembic_time_samples",
            "alembic_velocity",
        ] {
            let key = storage_key(name);
            assert_eq!(user_name(&key), Some(name), "round trip of {name}");
        }
        assert_eq!(storage_key("color_layers"), format!("{ESCAPE_PREFIX}color_layers"));
        assert_eq!(storage_key("skin_tone"), "skin_tone");
        assert_eq!(storage_key("user:x"), "user:user:x");
    }

    #[test]
    fn internal_keys_are_not_user_attributes() {
        assert_eq!(user_name("color_layers"), None);
        assert_eq!(user_name("custom_normal"), None);
        assert_eq!(user_name("alembic_velocity"), None);
        assert_eq!(user_name("alembic_time_samples"), None);
        assert_eq!(user_name("user:color_layers"), Some("color_layers"));
        assert_eq!(user_name("Col"), Some("Col"));
    }
}
