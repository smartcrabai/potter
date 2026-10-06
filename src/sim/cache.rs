use std::{
    collections::BTreeMap,
    fs::{self, File, OpenOptions},
    io::{Read, Write},
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
};

use glam::DMat4;
use serde_json::json;

use super::{SOLVER_VERSION, SimulationResult};
use crate::geom::Mesh;
use crate::{
    error::{ErrorCode, PotError, Result},
    hash,
    model::{Id, SceneDoc},
};

#[derive(serde::Deserialize, serde::Serialize)]
#[serde(deny_unknown_fields)]
struct ModifierSimulationCache {
    cache_key: String,
    mesh: Mesh,
    #[serde(default)]
    particles: Option<Vec<super::ParticleState>>,
}

pub(super) fn load_modifier_simulation(
    directory: &Path,
    key: &str,
) -> Result<Option<(Mesh, Option<Vec<super::ParticleState>>)>> {
    let path = cache_path(directory, key);
    let mut file = match File::open(path) {
        Ok(file) => file,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(PotError::io(&error)),
    };
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes)
        .map_err(|error| PotError::io(&error))?;
    let Ok(entry) = serde_json::from_slice::<ModifierSimulationCache>(&bytes) else {
        return Ok(None);
    };
    if entry.cache_key != key
        || entry.mesh.validate().is_err()
        || entry.particles.as_ref().is_some_and(|particles| {
            particles.iter().any(|particle| {
                particle
                    .position
                    .iter()
                    .chain(&particle.velocity)
                    .chain(&particle.normal)
                    .any(|value| !value.is_finite())
                    || !particle.birth_frame.is_finite()
                    || !particle.death_frame.is_finite()
                    || !particle.size.is_finite()
                    || particle.size < 0.0
                    || !particle.mass.is_finite()
                    || particle.mass <= 0.0
            })
        })
    {
        return Ok(None);
    }
    Ok(Some((entry.mesh, entry.particles)))
}

pub(super) fn store_modifier_simulation(
    directory: &Path,
    key: &str,
    mesh: &Mesh,
    particles: Option<&[super::ParticleState]>,
) -> Result<()> {
    let cache_directory = directory.join(".potter").join("cache");
    fs::create_dir_all(&cache_directory).map_err(|error| PotError::io(&error))?;
    let destination = cache_path(directory, key);
    let bytes = serde_json::to_vec(&ModifierSimulationCache {
        cache_key: key.to_owned(),
        mesh: mesh.clone(),
        particles: particles.map(<[super::ParticleState]>::to_vec),
    })
    .map_err(|error| {
        PotError::with_details(
            ErrorCode::EvaluationFailed,
            "could not serialize modifier simulation cache entry",
            json!({"cause": error.to_string()}),
        )
    })?;
    let (temporary_path, mut file) = create_temporary(&cache_directory)?;
    let write_result = file.write_all(&bytes).and_then(|()| file.sync_all());
    if let Err(error) = write_result {
        let _ = fs::remove_file(&temporary_path);
        return Err(PotError::io(&error));
    }
    drop(file);
    if let Err(error) = fs::rename(&temporary_path, &destination) {
        let _ = fs::remove_file(&temporary_path);
        return Err(PotError::io(&error));
    }
    if let Ok(directory_file) = File::open(&cache_directory) {
        let _ = directory_file.sync_all();
    }
    Ok(())
}

static TEMP_SEQUENCE: AtomicU64 = AtomicU64::new(0);

pub(super) fn cache_key(
    doc: &SceneDoc,
    scene_id: &Id,
    frame: f64,
    base_world_matrices: &BTreeMap<Id, DMat4>,
) -> Result<String> {
    if !frame.is_finite() {
        return Err(PotError::invalid_argument(
            "simulation frame must be finite",
        ));
    }
    if !doc.scenes.contains_key(scene_id) {
        return Err(PotError::new(
            ErrorCode::SceneNotFound,
            format!("scene `{scene_id}` was not found"),
        ));
    }

    let mut matrices = BTreeMap::new();
    for (id, matrix) in base_world_matrices {
        if !matrix.is_finite() {
            return Err(PotError::invalid_argument(format!(
                "base world matrix for `{id}` contains a non-finite value"
            )));
        }
        matrices.insert(id.to_string(), matrix.to_cols_array());
    }
    let snapshot = serde_json::to_value(doc).map_err(|error| {
        PotError::with_details(
            ErrorCode::InvalidArgument,
            "could not serialize the simulation input snapshot",
            json!({"cause": error.to_string()}),
        )
    })?;
    let input = json!({
        "solver_version": SOLVER_VERSION,
        "scene_id": scene_id.as_str(),
        "frame": frame,
        "scene_snapshot": snapshot,
        "base_world_matrices": matrices,
    });
    let canonical = hash::canonicalize(&input)?;
    Ok(hash::sha256(&canonical))
}

pub(super) fn load(directory: &Path, key: &str) -> Result<Option<SimulationResult>> {
    let path = cache_path(directory, key);
    let mut file = match File::open(path) {
        Ok(file) => file,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(PotError::io(&error)),
    };
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes)
        .map_err(|error| PotError::io(&error))?;
    let Ok(result) = serde_json::from_slice::<SimulationResult>(&bytes) else {
        return Ok(None);
    };
    if result.cache_key != key || !valid_result(&result, key) {
        return Ok(None);
    }
    Ok(Some(result))
}

fn valid_result(result: &SimulationResult, key: &str) -> bool {
    result
        .physics_cache_key
        .as_deref()
        .is_none_or(|physics_key| physics_key == key)
        && result
            .world_matrices
            .values()
            .all(|matrix| matrix.iter().all(|value| value.is_finite()))
        && result
            .linear_velocities
            .values()
            .all(|velocity| velocity.iter().all(|value| value.is_finite()))
        && result
            .deformed_vertices
            .values()
            .all(|vertices| vertices.iter().flatten().all(|value| value.is_finite()))
        && result
            .particle_positions
            .values()
            .all(|positions| positions.iter().flatten().all(|value| value.is_finite()))
        && result
            .particle_velocities
            .values()
            .all(|velocities| velocities.iter().flatten().all(|value| value.is_finite()))
        && result
            .particle_birth_frames
            .values()
            .flatten()
            .all(|value| value.is_finite())
        && result.particle_states.values().all(|states| {
            states.iter().all(|state| {
                state
                    .position
                    .iter()
                    .chain(&state.velocity)
                    .chain(&state.normal)
                    .all(|value| value.is_finite())
                    && state.birth_frame.is_finite()
                    && state.death_frame.is_finite()
                    && state.death_frame >= state.birth_frame
                    && state.size.is_finite()
                    && state.size >= 0.0
                    && state.mass.is_finite()
                    && state.mass > 0.0
            })
        })
        && result
            .fluid_particles
            .values()
            .all(|positions| positions.iter().flatten().all(|value| value.is_finite()))
        && result.fluid_surfaces.values().all(|mesh| {
            mesh.vertices.iter().all(|vertex| vertex.co.is_finite()) && mesh.validate().is_ok()
        })
        && result
            .paint_colors
            .values()
            .all(|colors| colors.iter().flatten().all(|value| value.is_finite()))
        && result
            .paint_weights
            .values()
            .all(|weights| weights.iter().all(|value| value.is_finite()))
        && result.particle_positions.iter().all(|(id, positions)| {
            result
                .particle_velocities
                .get(id)
                .is_some_and(|velocities| velocities.len() == positions.len())
                && result
                    .particle_birth_frames
                    .get(id)
                    .is_some_and(|birth_frames| birth_frames.len() == positions.len())
        })
}

pub(super) fn store(directory: &Path, result: &SimulationResult) -> Result<()> {
    let cache_directory = directory.join(".potter").join("cache");
    fs::create_dir_all(&cache_directory).map_err(|error| PotError::io(&error))?;
    let destination = cache_path(directory, &result.cache_key);
    let bytes = serde_json::to_vec(result).map_err(|error| {
        PotError::with_details(
            ErrorCode::EvaluationFailed,
            "could not serialize simulation cache entry",
            json!({"cause": error.to_string()}),
        )
    })?;

    let (temporary_path, mut file) = create_temporary(&cache_directory)?;
    let write_result = file.write_all(&bytes).and_then(|()| file.sync_all());
    if let Err(error) = write_result {
        let _ = fs::remove_file(&temporary_path);
        return Err(PotError::io(&error));
    }
    drop(file);
    if let Err(error) = fs::rename(&temporary_path, &destination) {
        let _ = fs::remove_file(&temporary_path);
        return Err(PotError::io(&error));
    }
    if let Ok(directory_file) = File::open(&cache_directory) {
        let _ = directory_file.sync_all();
    }
    Ok(())
}

fn cache_path(directory: &Path, key: &str) -> PathBuf {
    let digest = key.strip_prefix("sha256:").unwrap_or(key);
    directory
        .join(".potter")
        .join("cache")
        .join(format!("sha256-{digest}.json"))
}

fn create_temporary(directory: &Path) -> Result<(PathBuf, File)> {
    loop {
        let sequence = TEMP_SEQUENCE.fetch_add(1, Ordering::Relaxed);
        let path = directory.join(format!(".simulation-{}-{sequence}.tmp", std::process::id()));
        match OpenOptions::new().write(true).create_new(true).open(&path) {
            Ok(file) => return Ok((path, file)),
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {}
            Err(error) => return Err(PotError::io(&error)),
        }
    }
}
