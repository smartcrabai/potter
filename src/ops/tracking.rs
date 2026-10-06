use serde::de::DeserializeOwned;
use serde_json::{Map, Value};

use crate::{
    error::{ErrorCode, Result},
    model::{
        Id, MovieClip as StoredMovieClip, PlaneHomography, PlaneTrack, SolvedCamera,
        TrackingMarker, TrackingObject, TrackingObjectPose, TrackingTrack,
    },
    tracking::{
        CameraObservation, GrayFrame, KltConfig, LensDistortion, MovieClip, ObjectObservation,
        PointCorrespondence, solve_camera, solve_homography, solve_object_pose, track_marker,
    },
};

use super::{ChangeKind, Engine, check_fields, operation_pointer, parse_id, read_id, read_string};

pub(super) fn apply(
    engine: &mut Engine<'_>,
    name: &str,
    operation: &Map<String, Value>,
) -> Result<bool> {
    match name {
        "tracking.clip_create" => create_clip(engine, operation),
        "tracking.track_add" => add_track(engine, operation),
        "tracking.track" => track_markers(engine, operation),
        "tracking.plane_track" => create_plane_track(engine, operation),
        "tracking.solve_camera" => solve_camera_operation(engine, operation),
        "tracking.solve_object" => solve_object_operation(engine, operation),
        "tracking.set_camera_intrinsics" => set_camera_intrinsics(engine, operation),
        "tracking.undistort" => undistort_clip(engine, operation),
        _ => Err(engine.error(
            ErrorCode::InvalidOperation,
            format!("unsupported tracking operation `{name}`"),
            &operation_pointer(engine.operation_index, "op"),
        )),
    }
}

fn create_clip(engine: &mut Engine<'_>, operation: &Map<String, Value>) -> Result<bool> {
    check_fields(
        engine,
        operation,
        &[
            "op",
            "id",
            "name",
            "source",
            "source_hash",
            "frame_start",
            "fps",
            "width",
            "height",
        ],
        &["op", "id", "name"],
    )?;
    let id = read_id(engine, operation, "id")?;
    if engine.doc.movie_clips.contains_key(&id) {
        return Err(engine.error(
            ErrorCode::IdExists,
            "movie clip ID already exists",
            &operation_pointer(engine.operation_index, "id"),
        ));
    }
    let mut clip = StoredMovieClip {
        name: read_string(engine, operation, "name")?,
        ..StoredMovieClip::default()
    };
    clip.source = optional_string(engine, operation, "source")?;
    clip.source_hash = optional_string(engine, operation, "source_hash")?;
    clip.frame_start = optional_i32(engine, operation, "frame_start", clip.frame_start)?;
    clip.fps = optional_f64(engine, operation, "fps", clip.fps)?;
    clip.width =
        u32::try_from(optional_usize_field(engine, operation, "width", 0)?).map_err(|_| {
            engine.error(
                ErrorCode::InvalidArgument,
                "clip width exceeds u32",
                &operation_pointer(engine.operation_index, "width"),
            )
        })?;
    clip.height =
        u32::try_from(optional_usize_field(engine, operation, "height", 0)?).map_err(|_| {
            engine.error(
                ErrorCode::InvalidArgument,
                "clip height exceeds u32",
                &operation_pointer(engine.operation_index, "height"),
            )
        })?;
    validate_clip(engine, &clip, "")?;
    engine.doc.movie_clips.insert(id.clone(), clip);
    engine.mark("movie_clips", &id, ChangeKind::Created);
    Ok(true)
}

fn add_track(engine: &mut Engine<'_>, operation: &Map<String, Value>) -> Result<bool> {
    check_fields(
        engine,
        operation,
        &[
            "op",
            "id",
            "track",
            "name",
            "frame",
            "co",
            "pattern_corners",
            "search_area",
            "disabled",
        ],
        &["op", "id", "track", "name", "frame", "co"],
    )?;
    let clip_id = read_id(engine, operation, "id")?;
    let track_id = read_id(engine, operation, "track")?.to_string();
    let name = read_string(engine, operation, "name")?;
    let frame = required_f64(engine, operation, "frame")?;
    let co: [f64; 2] = deserialize_field(engine, operation, "co")?;
    let mut marker = TrackingMarker {
        frame,
        co,
        ..TrackingMarker::default()
    };
    if let Some(value) = operation.get("pattern_corners") {
        marker.pattern_corners = deserialize_value(engine, value, "pattern_corners")?;
    }
    if let Some(value) = operation.get("search_area") {
        marker.search_area = deserialize_value(engine, value, "search_area")?;
    }
    marker.disabled = optional_bool(engine, operation, "disabled", false)?;
    let mut clip = get_clip(engine, &clip_id)?;
    if clip
        .tracking
        .tracks
        .iter()
        .any(|track| track.id == track_id)
    {
        return Err(engine.error(
            ErrorCode::IdExists,
            "tracking track ID already exists on this clip",
            &operation_pointer(engine.operation_index, "track"),
        ));
    }
    clip.tracking.tracks.push(TrackingTrack {
        id: track_id,
        name,
        markers: vec![marker],
    });
    validate_clip(engine, &clip, "tracking")?;
    engine.doc.movie_clips.insert(clip_id.clone(), clip);
    engine.mark("movie_clips", &clip_id, ChangeKind::Updated);
    Ok(true)
}

fn track_markers(engine: &mut Engine<'_>, operation: &Map<String, Value>) -> Result<bool> {
    check_fields(
        engine,
        operation,
        &["op", "id", "track", "frames", "initial_frame", "config"],
        &["op", "id", "track", "frames"],
    )?;
    let clip_id = read_id(engine, operation, "id")?;
    let track_id = read_id(engine, operation, "track")?.to_string();
    let frames: Vec<GrayFrame> = deserialize_field(engine, operation, "frames")?;
    let mut clip = get_clip(engine, &clip_id)?;
    validate_clip(engine, &clip, "tracking")?;
    let track_index = clip
        .tracking
        .tracks
        .iter()
        .position(|track| track.id == track_id)
        .ok_or_else(|| {
            engine.error(
                ErrorCode::TargetNotFound,
                "tracking track does not exist",
                &operation_pointer(engine.operation_index, "track"),
            )
        })?;
    let track = &clip.tracking.tracks[track_index];
    let seed_frame = match operation.get("initial_frame") {
        Some(_) => required_i32(engine, operation, "initial_frame")?,
        None => track
            .markers
            .iter()
            .find(|marker| !marker.disabled)
            .map(|marker| frame_number(engine, marker.frame, "track/markers/frame"))
            .transpose()?
            .ok_or_else(|| {
                engine.error(
                    ErrorCode::InvalidOperation,
                    "tracking track has no enabled marker to use as a seed",
                    &operation_pointer(engine.operation_index, "track"),
                )
            })?,
    };
    let seed = track
        .markers
        .iter()
        .find(|marker| crate::float::equal_f64(marker.frame, f64::from(seed_frame)))
        .filter(|marker| !marker.disabled)
        .ok_or_else(|| {
            engine.error(
                ErrorCode::InvalidOperation,
                "tracking seed frame has no enabled marker",
                &operation_pointer(engine.operation_index, "initial_frame"),
            )
        })?;
    let config = read_klt_config(engine, operation)?;
    let runtime_clip = MovieClip::new(
        clip_id.to_string(),
        clip.name.clone(),
        clip.frame_start,
        frames,
    )
    .map_err(|error| {
        engine.error(
            ErrorCode::InvalidOperation,
            error.message,
            &operation_pointer(engine.operation_index, "frames"),
        )
    })?;
    let tracked = track_marker(&runtime_clip, seed_frame, seed.co, &config).map_err(|error| {
        engine.error(
            ErrorCode::InvalidOperation,
            error.message,
            &operation_pointer(engine.operation_index, "frames"),
        )
    })?;
    let first = f64::from(runtime_clip.frame_start);
    let final_offset = i32::try_from(runtime_clip.frame_count() - 1).map_err(|_| {
        engine.error(
            ErrorCode::InvalidOperation,
            "tracking frame range is out of bounds",
            &operation_pointer(engine.operation_index, "frames"),
        )
    })?;
    let last = f64::from(
        runtime_clip
            .frame_start
            .checked_add(final_offset)
            .ok_or_else(|| {
                engine.error(
                    ErrorCode::InvalidOperation,
                    "tracking frame range overflows",
                    &operation_pointer(engine.operation_index, "frames"),
                )
            })?,
    );
    let previous = &clip.tracking.tracks[track_index].markers;
    let mut markers: Vec<_> = previous
        .iter()
        .filter(|marker| marker.frame < first || marker.frame > last)
        .cloned()
        .collect();
    for result in tracked {
        let frame = f64::from(result.frame);
        let mut marker = previous
            .binary_search_by(|marker| marker.frame.total_cmp(&frame))
            .ok()
            .map_or_else(
                || TrackingMarker {
                    frame,
                    co: result.position,
                    ..TrackingMarker::default()
                },
                |index| previous[index].clone(),
            );
        if !marker.disabled {
            marker.co = result.position;
        }
        markers.push(marker);
    }
    markers.sort_by(|left, right| left.frame.total_cmp(&right.frame));
    if markers.as_slice() == previous {
        return Ok(false);
    }
    clip.tracking.tracks[track_index].markers = markers;
    if clip
        .tracking
        .plane_tracks
        .iter()
        .any(|plane_track| plane_track.track_ids.contains(&track_id))
    {
        refresh_plane_tracks(engine, &mut clip)?;
    }
    validate_clip(engine, &clip, "tracking")?;
    engine.doc.movie_clips.insert(clip_id.clone(), clip);
    engine.mark("movie_clips", &clip_id, ChangeKind::Updated);
    Ok(true)
}

fn create_plane_track(engine: &mut Engine<'_>, operation: &Map<String, Value>) -> Result<bool> {
    check_fields(
        engine,
        operation,
        &[
            "op",
            "id",
            "plane_track",
            "name",
            "track_ids",
            "reference_frame",
        ],
        &[
            "op",
            "id",
            "plane_track",
            "name",
            "track_ids",
            "reference_frame",
        ],
    )?;
    let clip_id = read_id(engine, operation, "id")?;
    let plane_track_id = read_id(engine, operation, "plane_track")?.to_string();
    let name = read_string(engine, operation, "name")?;
    let track_ids = operation
        .get("track_ids")
        .and_then(Value::as_array)
        .ok_or_else(|| {
            engine.error(
                ErrorCode::InvalidOperation,
                "track_ids must be an array",
                &operation_pointer(engine.operation_index, "track_ids"),
            )
        })?;
    if track_ids.len() < 4 {
        return Err(engine.error(
            ErrorCode::InvalidOperation,
            "plane tracking requires at least four tracks",
            &operation_pointer(engine.operation_index, "track_ids"),
        ));
    }
    let mut parsed_track_ids = Vec::with_capacity(track_ids.len());
    for (index, value) in track_ids.iter().enumerate() {
        let text = value.as_str().ok_or_else(|| {
            engine.error(
                ErrorCode::InvalidOperation,
                "track_ids items must be strings",
                &operation_pointer(engine.operation_index, &format!("track_ids/{index}")),
            )
        })?;
        parse_id(
            engine,
            text,
            &operation_pointer(engine.operation_index, &format!("track_ids/{index}")),
        )?;
        if parsed_track_ids.iter().any(|existing| existing == text) {
            return Err(engine.error(
                ErrorCode::InvalidOperation,
                "plane track IDs must not contain duplicates",
                &operation_pointer(engine.operation_index, "track_ids"),
            ));
        }
        parsed_track_ids.push(text.to_owned());
    }
    let reference_frame = required_f64(engine, operation, "reference_frame")?;
    let mut clip = get_clip(engine, &clip_id)?;
    if clip
        .tracking
        .plane_tracks
        .iter()
        .any(|plane_track| plane_track.id == plane_track_id)
    {
        return Err(engine.error(
            ErrorCode::IdExists,
            "plane track ID already exists on this clip",
            &operation_pointer(engine.operation_index, "plane_track"),
        ));
    }
    for track_id in &parsed_track_ids {
        if !clip
            .tracking
            .tracks
            .iter()
            .any(|track| &track.id == track_id)
        {
            return Err(engine.error(
                ErrorCode::TargetNotFound,
                format!("tracking track `{track_id}` does not exist"),
                &operation_pointer(engine.operation_index, "track_ids"),
            ));
        }
    }
    let homographies = build_plane_homographies(
        engine,
        &clip.tracking.tracks,
        &parsed_track_ids,
        reference_frame,
    )?;
    clip.tracking.plane_tracks.push(PlaneTrack {
        id: plane_track_id,
        name,
        track_ids: parsed_track_ids,
        reference_frame,
        homographies,
    });
    validate_clip(engine, &clip, "tracking")?;
    engine.doc.movie_clips.insert(clip_id.clone(), clip);
    engine.mark("movie_clips", &clip_id, ChangeKind::Updated);
    Ok(true)
}

fn refresh_plane_tracks(engine: &Engine<'_>, clip: &mut StoredMovieClip) -> Result<()> {
    let (tracks, plane_tracks) = (&clip.tracking.tracks, &mut clip.tracking.plane_tracks);
    for plane_track in plane_tracks {
        plane_track.homographies = build_plane_homographies(
            engine,
            tracks,
            &plane_track.track_ids,
            plane_track.reference_frame,
        )?;
    }
    Ok(())
}

fn build_plane_homographies(
    engine: &Engine<'_>,
    tracks: &[TrackingTrack],
    track_ids: &[String],
    reference_frame: f64,
) -> Result<Vec<PlaneHomography>> {
    if track_ids.len() < 4 {
        return Err(engine.error(
            ErrorCode::InvalidOperation,
            "plane tracking requires at least four tracks",
            &operation_pointer(engine.operation_index, "track_ids"),
        ));
    }
    let reference_positions: Vec<_> = track_ids
        .iter()
        .map(|track_id| {
            tracks
                .iter()
                .find(|track| &track.id == track_id)
                .ok_or_else(|| {
                    engine.error(
                        ErrorCode::TargetNotFound,
                        format!("tracking track `{track_id}` does not exist"),
                        &operation_pointer(engine.operation_index, "track_ids"),
                    )
                })?
                .markers
                .iter()
                .find(|marker| {
                    crate::float::equal_f64(marker.frame, reference_frame) && !marker.disabled
                })
                .map(|marker| marker.co)
                .ok_or_else(|| {
                    engine.error(
                        ErrorCode::InvalidOperation,
                        "every plane track needs an enabled reference marker",
                        &operation_pointer(engine.operation_index, "reference_frame"),
                    )
                })
        })
        .collect::<Result<_>>()?;
    let mut candidate_frames = Vec::new();
    for track_id in track_ids {
        let track = tracks
            .iter()
            .find(|track| &track.id == track_id)
            .ok_or_else(|| {
                engine.error(
                    ErrorCode::TargetNotFound,
                    format!("tracking track `{track_id}` does not exist"),
                    &operation_pointer(engine.operation_index, "track_ids"),
                )
            })?;
        candidate_frames.extend(
            track
                .markers
                .iter()
                .filter(|marker| !marker.disabled)
                .map(|marker| marker.frame),
        );
    }
    candidate_frames.sort_by(f64::total_cmp);
    candidate_frames.dedup_by(|left, right| crate::float::equal_f64(*left, *right));
    let mut homographies = Vec::new();
    for frame in candidate_frames {
        let mut correspondences = Vec::with_capacity(track_ids.len());
        for (index, track_id) in track_ids.iter().enumerate() {
            let destination = tracks
                .iter()
                .find(|track| &track.id == track_id)
                .and_then(|track| {
                    track.markers.iter().find(|marker| {
                        crate::float::equal_f64(marker.frame, frame) && !marker.disabled
                    })
                });
            if let Some(destination) = destination {
                correspondences.push(PointCorrespondence {
                    source: reference_positions[index],
                    destination: destination.co,
                });
            }
        }
        if correspondences.len() >= 4 {
            let homography = solve_homography(&correspondences).map_err(|error| {
                engine.error(
                    ErrorCode::InvalidOperation,
                    error.message,
                    &operation_pointer(engine.operation_index, "track_ids"),
                )
            })?;
            homographies.push(PlaneHomography {
                frame,
                matrix: homography.matrix,
            });
        }
    }
    if homographies.is_empty() {
        return Err(engine.error(
            ErrorCode::InvalidOperation,
            "no frame has four usable plane track markers",
            &operation_pointer(engine.operation_index, "track_ids"),
        ));
    }
    Ok(homographies)
}

fn solve_camera_operation(engine: &mut Engine<'_>, operation: &Map<String, Value>) -> Result<bool> {
    check_fields(
        engine,
        operation,
        &["op", "id", "frame", "observations"],
        &["op", "id", "frame", "observations"],
    )?;
    let clip_id = read_id(engine, operation, "id")?;
    let frame = required_i32(engine, operation, "frame")?;
    let observations: Vec<CameraObservation> =
        deserialize_field(engine, operation, "observations")?;
    let mut clip = get_clip(engine, &clip_id)?;
    let camera = solve_camera(&observations).map_err(|error| {
        engine.error(
            ErrorCode::InvalidOperation,
            error.message,
            &operation_pointer(engine.operation_index, "observations"),
        )
    })?;
    let solved = SolvedCamera {
        frame,
        matrix: camera.matrix,
        average_error: 0.0,
        matrix_is_camera_to_world: false,
    };
    if let Some(existing) = clip
        .tracking
        .reconstruction
        .cameras
        .iter_mut()
        .find(|existing| existing.frame == frame)
    {
        if *existing == solved {
            return Ok(false);
        }
        *existing = solved;
    } else {
        clip.tracking.reconstruction.cameras.push(solved);
        clip.tracking
            .reconstruction
            .cameras
            .sort_by_key(|camera| camera.frame);
    }
    clip.tracking.reconstruction.is_valid = true;
    validate_clip(engine, &clip, "tracking/reconstruction")?;
    engine.doc.movie_clips.insert(clip_id.clone(), clip);
    engine.mark("movie_clips", &clip_id, ChangeKind::Updated);
    Ok(true)
}
fn solve_object_operation(engine: &mut Engine<'_>, operation: &Map<String, Value>) -> Result<bool> {
    check_fields(
        engine,
        operation,
        &["op", "id", "object", "name", "tracks"],
        &["op", "id", "object", "tracks"],
    )?;
    let clip_id = read_id(engine, operation, "id")?;
    let object_id = read_string(engine, operation, "object")?;
    Id::new(object_id.clone()).map_err(|error| {
        engine.error(
            ErrorCode::InvalidArgument,
            error.message,
            &operation_pointer(engine.operation_index, "object"),
        )
    })?;
    let name = operation
        .get("name")
        .map(|value| {
            value
                .as_str()
                .filter(|name| !name.is_empty())
                .map(str::to_owned)
                .ok_or_else(|| {
                    engine.error(
                        ErrorCode::InvalidArgument,
                        "tracking object name must be a non-empty string",
                        &operation_pointer(engine.operation_index, "name"),
                    )
                })
        })
        .transpose()?
        .unwrap_or_else(|| object_id.clone());
    let track_ids: Vec<String> = deserialize_field(engine, operation, "tracks")?;
    if track_ids.len() < 4 {
        return Err(engine.error(
            ErrorCode::InvalidOperation,
            "object solve requires at least four tracking tracks",
            &operation_pointer(engine.operation_index, "tracks"),
        ));
    }
    let unique_tracks = track_ids.iter().collect::<std::collections::BTreeSet<_>>();
    if unique_tracks.len() != track_ids.len() {
        return Err(engine.error(
            ErrorCode::InvalidOperation,
            "object solve track IDs must be unique",
            &operation_pointer(engine.operation_index, "tracks"),
        ));
    }
    let mut clip = get_clip(engine, &clip_id)?;
    for track_id in &track_ids {
        if !clip
            .tracking
            .tracks
            .iter()
            .any(|track| track.id == *track_id)
            || !clip
                .tracking
                .reconstruction
                .points
                .iter()
                .any(|point| point.track == *track_id)
        {
            return Err(engine.error(
                ErrorCode::TargetNotFound,
                format!("object solve track `{track_id}` needs a reconstructed 3D point"),
                &operation_pointer(engine.operation_index, "tracks"),
            ));
        }
    }
    let mut cameras = clip.tracking.reconstruction.cameras.clone();
    cameras.sort_by_key(|camera| camera.frame);
    let mut poses = Vec::with_capacity(cameras.len());
    let initial_scale = clip
        .tracking
        .objects
        .iter()
        .find(|object| object.id == object_id)
        .map_or(1.0, |object| object.scale);
    let mut initial = glam::DMat4::from_scale_rotation_translation(
        glam::DVec3::splat(initial_scale),
        glam::DQuat::IDENTITY,
        glam::DVec3::ZERO,
    )
    .to_cols_array();
    let mut object_scale = None;
    for camera in cameras {
        if poses.last().is_some_and(|pose: &TrackingObjectPose| {
            crate::float::equal_f64(pose.frame, f64::from(camera.frame))
        }) {
            return Err(engine.error(
                ErrorCode::InvalidOperation,
                "object solve requires unique solved-camera frames",
                &operation_pointer(engine.operation_index, "id"),
            ));
        }
        let frame = f64::from(camera.frame);
        let mut observations = Vec::with_capacity(track_ids.len());
        for track_id in &track_ids {
            let track = clip
                .tracking
                .tracks
                .iter()
                .find(|track| track.id == *track_id)
                .ok_or_else(|| {
                    engine.error(
                        ErrorCode::InternalError,
                        "validated tracking track disappeared",
                        &operation_pointer(engine.operation_index, "tracks"),
                    )
                })?;
            let marker = track
                .markers
                .iter()
                .find(|marker| crate::float::equal_f64(marker.frame, frame) && !marker.disabled);
            let Some(marker) = marker else {
                continue;
            };
            let point = clip
                .tracking
                .reconstruction
                .points
                .iter()
                .find(|point| point.track == *track_id)
                .ok_or_else(|| {
                    engine.error(
                        ErrorCode::InternalError,
                        "validated reconstructed point disappeared",
                        &operation_pointer(engine.operation_index, "tracks"),
                    )
                })?;
            observations.push(ObjectObservation {
                object: point.co,
                image: marker.co,
            });
        }
        if observations.len() < 4 {
            return Err(engine.error(
                ErrorCode::InvalidOperation,
                format!("object solve has fewer than four enabled markers at frame {frame}"),
                &operation_pointer(engine.operation_index, "tracks"),
            ));
        }
        let camera_model = crate::tracking::CameraModel {
            matrix: camera.matrix,
        };
        let solved = solve_object_pose(&camera_model, &observations, initial).map_err(|error| {
            engine.error(
                ErrorCode::InvalidOperation,
                error.message,
                &operation_pointer(engine.operation_index, "tracks"),
            )
        })?;
        let camera_world = crate::tracking::camera_world_matrix(
            &camera_model,
            &clip.tracking.camera,
            clip.width,
            clip.height,
        )
        .map_err(|error| {
            engine.error(
                ErrorCode::InvalidOperation,
                error.message,
                &operation_pointer(engine.operation_index, "id"),
            )
        })?;
        let camera_world = glam::DMat4::from_cols_array(&camera_world);
        let object_world = glam::DMat4::from_cols_array(&solved.matrix);
        let object_to_camera = camera_world.inverse() * object_world;
        let camera_to_object = object_to_camera.inverse();
        if !camera_to_object
            .to_cols_array()
            .iter()
            .all(|value| value.is_finite())
        {
            return Err(engine.error(
                ErrorCode::InvalidOperation,
                "object solve produced a non-finite camera-relative pose",
                &operation_pointer(engine.operation_index, "tracks"),
            ));
        }
        initial = solved.matrix;
        object_scale.get_or_insert(solved.scale);
        poses.push(TrackingObjectPose {
            frame,
            matrix: camera_to_object.to_cols_array(),
            average_error: 0.0,
        });
    }
    if poses.is_empty() {
        return Err(engine.error(
            ErrorCode::InvalidOperation,
            "object solve requires at least one solved-camera frame",
            &operation_pointer(engine.operation_index, "id"),
        ));
    }
    let object = TrackingObject {
        id: object_id.clone(),
        name,
        tracks: track_ids,
        reconstruction: poses,
        reconstruction_is_valid: true,
        reconstruction_average_error: 0.0,
        scale: object_scale.unwrap_or(1.0),
    };
    let existing = clip
        .tracking
        .objects
        .iter()
        .position(|existing| existing.id == object_id);
    if existing.is_some_and(|index| clip.tracking.objects[index] == object) {
        return Ok(false);
    }
    if let Some(index) = existing {
        clip.tracking.objects[index] = object;
    } else {
        clip.tracking.objects.push(object);
        clip.tracking
            .objects
            .sort_by(|left, right| left.id.cmp(&right.id));
    }
    validate_clip(engine, &clip, "tracking")?;
    engine.doc.movie_clips.insert(clip_id.clone(), clip);
    engine.mark("movie_clips", &clip_id, ChangeKind::Updated);
    Ok(true)
}

fn set_camera_intrinsics(engine: &mut Engine<'_>, operation: &Map<String, Value>) -> Result<bool> {
    check_fields(
        engine,
        operation,
        &["op", "id", "set"],
        &["op", "id", "set"],
    )?;
    let clip_id = read_id(engine, operation, "id")?;
    let set = operation
        .get("set")
        .and_then(Value::as_object)
        .ok_or_else(|| {
            engine.error(
                ErrorCode::InvalidOperation,
                "set must be an object",
                &operation_pointer(engine.operation_index, "set"),
            )
        })?;
    if set.is_empty() {
        return Err(engine.error(
            ErrorCode::InvalidOperation,
            "set_camera_intrinsics set must not be empty",
            &operation_pointer(engine.operation_index, "set"),
        ));
    }
    let mut clip = get_clip(engine, &clip_id)?;
    let previous = clip.tracking.camera.clone();
    for (key, value) in set {
        let pointer = operation_pointer(engine.operation_index, &format!("set/{key}"));
        match key.as_str() {
            "focal_mm" => {
                clip.tracking.camera.focal_mm = read_f64_value(engine, value, &pointer)?;
            }
            "sensor_width_mm" => {
                clip.tracking.camera.sensor_width_mm = read_f64_value(engine, value, &pointer)?;
            }
            "principal" => {
                clip.tracking.camera.principal = deserialize_value(engine, value, &pointer)?;
            }
            "k1" => clip.tracking.camera.k1 = read_f64_value(engine, value, &pointer)?,
            "k2" => clip.tracking.camera.k2 = read_f64_value(engine, value, &pointer)?,
            "k3" => clip.tracking.camera.k3 = read_f64_value(engine, value, &pointer)?,
            _ => {
                return Err(engine.error(
                    ErrorCode::InvalidOperation,
                    format!("unknown camera intrinsic field `{key}`"),
                    &pointer,
                ));
            }
        }
    }
    clip.tracking.camera.validate().map_err(|error| {
        engine.error(
            ErrorCode::InvalidOperation,
            error.message,
            &operation_pointer(engine.operation_index, "set"),
        )
    })?;
    if clip.tracking.camera == previous {
        return Ok(false);
    }
    validate_clip(engine, &clip, "tracking/camera")?;
    engine.doc.movie_clips.insert(clip_id.clone(), clip);
    engine.mark("movie_clips", &clip_id, ChangeKind::Updated);
    Ok(true)
}

fn undistort_clip(engine: &mut Engine<'_>, operation: &Map<String, Value>) -> Result<bool> {
    check_fields(
        engine,
        operation,
        &["op", "id", "width", "height"],
        &["op", "id", "width", "height"],
    )?;
    let clip_id = read_id(engine, operation, "id")?;
    let width = required_usize(engine, operation, "width")?;
    let height = required_usize(engine, operation, "height")?;
    if width == 0 || height == 0 {
        return Err(engine.error(
            ErrorCode::InvalidOperation,
            "undistortion image dimensions must be non-zero",
            &operation_pointer(engine.operation_index, "width"),
        ));
    }
    let mut clip = get_clip(engine, &clip_id)?;
    let camera = &clip.tracking.camera;
    let width_f64 = f64::from(u32::try_from(width).map_err(|_| {
        engine.error(
            ErrorCode::InvalidOperation,
            "undistortion image width exceeds supported range",
            &operation_pointer(engine.operation_index, "width"),
        )
    })?);
    let height_f64 = f64::from(u32::try_from(height).map_err(|_| {
        engine.error(
            ErrorCode::InvalidOperation,
            "undistortion image height exceeds supported range",
            &operation_pointer(engine.operation_index, "height"),
        )
    })?);
    // Camera principal coordinates are normalized; lens distortion operates in pixels.
    let lens = LensDistortion {
        k1: camera.k1,
        k2: camera.k2,
        k3: camera.k3,
        p1: 0.0,
        p2: 0.0,
        center: [
            camera.principal[0] * width_f64,
            camera.principal[1] * height_f64,
        ],
        scale: [width_f64 * 0.5, height_f64 * 0.5],
    };
    lens.validate().map_err(|error| {
        engine.error(
            ErrorCode::InvalidOperation,
            error.message,
            &operation_pointer(engine.operation_index, "id"),
        )
    })?;
    let mut changed = false;
    for track in &mut clip.tracking.tracks {
        for marker in &mut track.markers {
            let original = marker.co;
            marker.co = lens.undistort(marker.co).map_err(|error| {
                engine.error(
                    ErrorCode::InvalidOperation,
                    error.message,
                    &operation_pointer(engine.operation_index, "id"),
                )
            })?;
            changed |= !crate::float::equal_f64_array(&marker.co, &original);
            for corner in &mut marker.pattern_corners {
                let original = *corner;
                *corner = lens.undistort(*corner).map_err(|error| {
                    engine.error(
                        ErrorCode::InvalidOperation,
                        error.message,
                        &operation_pointer(engine.operation_index, "id"),
                    )
                })?;
                changed |= !crate::float::equal_f64_array(corner, &original);
            }
            for corner in &mut marker.search_area {
                let original = *corner;
                *corner = lens.undistort(*corner).map_err(|error| {
                    engine.error(
                        ErrorCode::InvalidOperation,
                        error.message,
                        &operation_pointer(engine.operation_index, "id"),
                    )
                })?;
                changed |= !crate::float::equal_f64_array(corner, &original);
            }
        }
    }
    if !changed {
        return Ok(false);
    }
    refresh_plane_tracks(engine, &mut clip)?;
    validate_clip(engine, &clip, "tracking")?;
    engine.doc.movie_clips.insert(clip_id.clone(), clip);
    engine.mark("movie_clips", &clip_id, ChangeKind::Updated);
    Ok(true)
}

fn read_klt_config(engine: &Engine<'_>, operation: &Map<String, Value>) -> Result<KltConfig> {
    let mut config = KltConfig::default();
    let Some(value) = operation.get("config") else {
        return Ok(config);
    };
    let object = value.as_object().ok_or_else(|| {
        engine.error(
            ErrorCode::InvalidOperation,
            "config must be an object",
            &operation_pointer(engine.operation_index, "config"),
        )
    })?;
    for key in object.keys() {
        if ![
            "patch_radius",
            "max_iterations",
            "max_pyramid_levels",
            "convergence_threshold",
            "min_eigenvalue",
        ]
        .contains(&key.as_str())
        {
            return Err(engine.error(
                ErrorCode::InvalidOperation,
                format!("unknown KLT config field `{key}`"),
                &operation_pointer(engine.operation_index, &format!("config/{key}")),
            ));
        }
    }
    config.patch_radius =
        optional_usize_field(engine, object, "patch_radius", config.patch_radius)?;
    config.max_iterations =
        optional_usize_field(engine, object, "max_iterations", config.max_iterations)?;
    config.max_pyramid_levels = optional_usize_field(
        engine,
        object,
        "max_pyramid_levels",
        config.max_pyramid_levels,
    )?;
    config.convergence_threshold = optional_f64_field(
        engine,
        object,
        "convergence_threshold",
        config.convergence_threshold,
    )?;
    config.min_eigenvalue =
        optional_f64_field(engine, object, "min_eigenvalue", config.min_eigenvalue)?;
    Ok(config)
}

fn get_clip(engine: &Engine<'_>, id: &Id) -> Result<StoredMovieClip> {
    engine.doc.movie_clips.get(id).cloned().ok_or_else(|| {
        engine.error(
            ErrorCode::TargetNotFound,
            "movie clip does not exist",
            &operation_pointer(engine.operation_index, "id"),
        )
    })
}

fn validate_clip(engine: &Engine<'_>, clip: &StoredMovieClip, field: &str) -> Result<()> {
    clip.validate().map_err(|error| {
        engine.error(
            ErrorCode::InvalidOperation,
            error.message,
            &operation_pointer(engine.operation_index, field),
        )
    })
}

fn deserialize_field<T: DeserializeOwned>(
    engine: &Engine<'_>,
    object: &Map<String, Value>,
    key: &str,
) -> Result<T> {
    let value = object.get(key).ok_or_else(|| {
        engine.error(
            ErrorCode::InvalidOperation,
            format!("{key} is required"),
            &operation_pointer(engine.operation_index, key),
        )
    })?;
    deserialize_value(engine, value, key)
}

fn deserialize_value<T: DeserializeOwned>(
    engine: &Engine<'_>,
    value: &Value,
    field: &str,
) -> Result<T> {
    serde_json::from_value(value.clone()).map_err(|error| {
        engine.error(
            ErrorCode::InvalidOperation,
            format!("invalid {field}: {error}"),
            &operation_pointer(engine.operation_index, field),
        )
    })
}

fn optional_string(
    engine: &Engine<'_>,
    operation: &Map<String, Value>,
    key: &str,
) -> Result<Option<String>> {
    match operation.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(value)) if !value.trim().is_empty() => Ok(Some(value.clone())),
        Some(_) => Err(engine.error(
            ErrorCode::InvalidOperation,
            format!("{key} must be a non-empty string or null"),
            &operation_pointer(engine.operation_index, key),
        )),
    }
}

fn optional_i32(
    engine: &Engine<'_>,
    operation: &Map<String, Value>,
    key: &str,
    default: i32,
) -> Result<i32> {
    match operation.get(key) {
        None => Ok(default),
        Some(value) => value
            .as_i64()
            .and_then(|number| i32::try_from(number).ok())
            .ok_or_else(|| {
                engine.error(
                    ErrorCode::InvalidOperation,
                    format!("{key} must be a 32-bit integer"),
                    &operation_pointer(engine.operation_index, key),
                )
            }),
    }
}

fn required_i32(engine: &Engine<'_>, operation: &Map<String, Value>, key: &str) -> Result<i32> {
    operation
        .get(key)
        .and_then(Value::as_i64)
        .and_then(|number| i32::try_from(number).ok())
        .ok_or_else(|| {
            engine.error(
                ErrorCode::InvalidOperation,
                format!("{key} must be a 32-bit integer"),
                &operation_pointer(engine.operation_index, key),
            )
        })
}

fn required_f64(engine: &Engine<'_>, operation: &Map<String, Value>, key: &str) -> Result<f64> {
    let value = operation.get(key).and_then(Value::as_f64).ok_or_else(|| {
        engine.error(
            ErrorCode::InvalidOperation,
            format!("{key} must be a number"),
            &operation_pointer(engine.operation_index, key),
        )
    })?;
    if value.is_finite() {
        Ok(value)
    } else {
        Err(engine.error(
            ErrorCode::InvalidOperation,
            format!("{key} must be finite"),
            &operation_pointer(engine.operation_index, key),
        ))
    }
}

fn optional_f64(
    engine: &Engine<'_>,
    operation: &Map<String, Value>,
    key: &str,
    default: f64,
) -> Result<f64> {
    match operation.get(key) {
        None => Ok(default),
        Some(value) => read_f64_value(
            engine,
            value,
            &operation_pointer(engine.operation_index, key),
        ),
    }
}

fn read_f64_value(engine: &Engine<'_>, value: &Value, pointer: &str) -> Result<f64> {
    value
        .as_f64()
        .filter(|number| number.is_finite())
        .ok_or_else(|| {
            engine.error(
                ErrorCode::InvalidOperation,
                "value must be a finite number",
                pointer,
            )
        })
}

fn required_usize(engine: &Engine<'_>, operation: &Map<String, Value>, key: &str) -> Result<usize> {
    operation
        .get(key)
        .and_then(Value::as_u64)
        .and_then(|number| usize::try_from(number).ok())
        .ok_or_else(|| {
            engine.error(
                ErrorCode::InvalidOperation,
                format!("{key} must be a non-negative integer"),
                &operation_pointer(engine.operation_index, key),
            )
        })
}

fn optional_usize_field(
    engine: &Engine<'_>,
    object: &Map<String, Value>,
    key: &str,
    default: usize,
) -> Result<usize> {
    object.get(key).map_or(Ok(default), |value| {
        value
            .as_u64()
            .and_then(|number| usize::try_from(number).ok())
            .ok_or_else(|| {
                engine.error(
                    ErrorCode::InvalidOperation,
                    format!("config/{key} must be a non-negative integer"),
                    &operation_pointer(engine.operation_index, &format!("config/{key}")),
                )
            })
    })
}

fn optional_f64_field(
    engine: &Engine<'_>,
    object: &Map<String, Value>,
    key: &str,
    default: f64,
) -> Result<f64> {
    object.get(key).map_or(Ok(default), |value| {
        read_f64_value(
            engine,
            value,
            &operation_pointer(engine.operation_index, &format!("config/{key}")),
        )
    })
}

fn optional_bool(
    engine: &Engine<'_>,
    operation: &Map<String, Value>,
    key: &str,
    default: bool,
) -> Result<bool> {
    match operation.get(key) {
        None => Ok(default),
        Some(value) => value.as_bool().ok_or_else(|| {
            engine.error(
                ErrorCode::InvalidOperation,
                format!("{key} must be a boolean"),
                &operation_pointer(engine.operation_index, key),
            )
        }),
    }
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "frame_number checks integer and i32 bounds before conversion"
)]
fn frame_number(engine: &Engine<'_>, value: f64, field: &str) -> Result<i32> {
    if value.fract() != 0.0 || value < f64::from(i32::MIN) || value > f64::from(i32::MAX) {
        return Err(engine.error(
            ErrorCode::InvalidOperation,
            "tracking frame must be an integer in the supported range",
            &operation_pointer(engine.operation_index, field),
        ));
    }
    Ok(value as i32)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{model::SceneDoc, ops::apply_batch};
    use serde_json::json;

    #[test]
    fn tracking_operations_create_and_track_a_marker_over_supplied_frames() -> Result<()> {
        let document = SceneDoc::new("tracking-test".to_owned());
        let frames = vec![make_texture(48, 48, 0.0)?, make_texture(48, 48, 1.0)?];
        let operations = json!({
            "schema_version": 1,
            "base_revision": 0,
            "operations": [
                {"op":"tracking.clip_create","id":"clip","name":"Test","frame_start":1},
                {"op":"tracking.track_add","id":"clip","track":"corner","name":"Corner","frame":1,"co":[24.0,24.0]},
                {"op":"tracking.track","id":"clip","track":"corner","frames":frames}
            ]
        });
        let outcome = apply_batch(&document, &operations)?;
        let clip = &outcome.doc.movie_clips[&Id::new("clip".to_owned())?];
        assert_eq!(clip.tracking.tracks[0].markers.len(), 2);
        assert!((clip.tracking.tracks[0].markers[1].co[0] - 25.0).abs() < 0.4);
        assert!((clip.tracking.tracks[0].markers[1].co[1] - 24.0).abs() < 0.4);
        Ok(())
    }

    #[test]
    fn tracking_operation_rejects_a_degenerate_plane_track_and_invalid_intrinsics() -> Result<()> {
        let document = SceneDoc::new("tracking-test".to_owned());
        let create = json!({
            "schema_version":1,"base_revision":0,
            "operations":[{"op":"tracking.clip_create","id":"clip","name":"Test"}]
        });
        let document = apply_batch(&document, &create)?.doc;
        let invalid_intrinsics = json!({
            "schema_version":1,"base_revision":1,
            "operations":[{"op":"tracking.set_camera_intrinsics","id":"clip","set":{"focal_mm":0.0}}]
        });
        assert!(apply_batch(&document, &invalid_intrinsics).is_err());
        let track_ops = json!({
            "schema_version":1,"base_revision":1,
            "operations":[
                {"op":"tracking.track_add","id":"clip","track":"a","name":"A","frame":1,"co":[0.0,0.0]},
                {"op":"tracking.track_add","id":"clip","track":"b","name":"B","frame":1,"co":[1.0,1.0]},
                {"op":"tracking.track_add","id":"clip","track":"c","name":"C","frame":1,"co":[2.0,2.0]},
                {"op":"tracking.track_add","id":"clip","track":"d","name":"D","frame":1,"co":[3.0,3.0]},
                {"op":"tracking.plane_track","id":"clip","plane_track":"plane","name":"Plane","track_ids":["a","b","c","d"],"reference_frame":1}
            ]
        });
        assert!(apply_batch(&document, &track_ops).is_err());
        Ok(())
    }

    #[test]
    fn plane_track_uses_the_tracked_markers_to_fit_each_frame() -> Result<()> {
        let document = SceneDoc::new("tracking-test".to_owned());
        let frames = json!([make_texture(48, 48, 0.0)?, make_texture(48, 48, 1.0)?]);
        let points = [
            ("top_left", [15.0, 15.0]),
            ("top_right", [33.0, 15.0]),
            ("bottom_left", [15.0, 33.0]),
            ("bottom_right", [33.0, 33.0]),
        ];
        let mut operations = vec![json!({
            "op":"tracking.clip_create","id":"clip","name":"Test","frame_start":1
        })];
        for (track, co) in points {
            operations.push(json!({
                "op":"tracking.track_add",
                "id":"clip",
                "track":track,
                "name":track,
                "frame":1,
                "co":co
            }));
        }
        for (track, _) in points {
            operations.push(json!({
                "op":"tracking.track","id":"clip","track":track,"frames":frames.clone()
            }));
        }
        operations.push(json!({
            "op":"tracking.plane_track",
            "id":"clip",
            "plane_track":"plane",
            "name":"Plane",
            "track_ids":["top_left","top_right","bottom_left","bottom_right"],
            "reference_frame":1
        }));
        operations.push(json!({
            "op":"tracking.set_camera_intrinsics",
            "id":"clip",
            "set":{"k1":0.1,"k2":-0.01,"k3":0.001}
        }));
        operations.push(json!({
            "op":"tracking.undistort","id":"clip","width":48,"height":48
        }));
        let outcome = apply_batch(
            &document,
            &json!({"schema_version":1,"base_revision":0,"operations":operations}),
        )?;
        let clip = &outcome.doc.movie_clips[&Id::new("clip".to_owned())?];
        let plane = &clip.tracking.plane_tracks[0];
        assert_eq!(plane.homographies.len(), 2);
        let moved = plane
            .homographies
            .iter()
            .find(|homography| homography.frame == 2.0)
            .ok_or_else(|| crate::error::PotError::invalid_operation("frame homography missing"))?;
        let source = clip.tracking.tracks[0].markers[0].co;
        let target = clip.tracking.tracks[0].markers[1].co;
        let projected = crate::tracking::Homography {
            matrix: moved.matrix,
        }
        .transform(source)?;
        assert!((projected[0] - target[0]).abs() < 0.1);
        assert!((projected[1] - target[1]).abs() < 0.1);
        Ok(())
    }

    #[test]
    fn camera_solve_operation_persists_the_reprojecting_camera() -> Result<()> {
        let document = SceneDoc::new("tracking-test".to_owned());
        let world_points = [
            [-2.0, -1.0, 4.0],
            [1.0, -1.0, 5.0],
            [2.0, 2.0, 6.0],
            [-1.0, 3.0, 7.0],
            [3.0, -2.0, 8.0],
            [-3.0, 2.0, 9.0],
        ];
        let observations: Vec<_> = world_points
            .iter()
            .map(|world| {
                let denominator = world[2] + 1.0;
                CameraObservation {
                    world: *world,
                    image: [
                        (2.0 * world[0] + world[2]) / denominator,
                        (3.0 * world[1] + world[2]) / denominator,
                    ],
                }
            })
            .collect();
        let outcome = apply_batch(
            &document,
            &json!({
                "schema_version":1,
                "base_revision":0,
                "operations":[
                    {"op":"tracking.clip_create","id":"clip","name":"Test"},
                    {"op":"tracking.solve_camera","id":"clip","frame":1,"observations":observations}
                ]
            }),
        )?;
        let clip = &outcome.doc.movie_clips[&Id::new("clip".to_owned())?];
        let solved = &clip.tracking.reconstruction.cameras[0];
        let camera = crate::tracking::CameraModel {
            matrix: solved.matrix,
        };
        let projected = camera.project(world_points[0])?;
        assert!((projected[0] - observations[0].image[0]).abs() < 1e-7);
        assert!((projected[1] - observations[0].image[1]).abs() < 1e-7);
        Ok(())
    }

    #[test]
    fn undistort_operation_updates_markers_using_radial_intrinsics() -> Result<()> {
        let document = SceneDoc::new("tracking-test".to_owned());
        let operations = json!({
            "schema_version":1,
            "base_revision":0,
            "operations":[
                {"op":"tracking.clip_create","id":"clip","name":"Test"},
                {"op":"tracking.set_camera_intrinsics","id":"clip","set":{"k1":0.1,"k2":-0.01,"k3":0.001}},
                {"op":"tracking.track_add","id":"clip","track":"point","name":"Point","frame":1,"co":[50.0,40.0]},
                {"op":"tracking.undistort","id":"clip","width":200,"height":160}
            ]
        });
        let outcome = apply_batch(&document, &operations)?;
        let clip = &outcome.doc.movie_clips[&Id::new("clip".to_owned())?];
        let coordinate = clip.tracking.tracks[0].markers[0].co;
        assert!(coordinate[0] > 50.0 && coordinate[0] < 100.0);
        assert!(coordinate[1] > 40.0 && coordinate[1] < 80.0);
        Ok(())
    }

    #[expect(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        reason = "generated grayscale test frames use bounded pixel indices"
    )]
    fn make_texture(width: usize, height: usize, shift_x: f64) -> Result<GrayFrame> {
        let pixels = (0..height)
            .flat_map(|y| {
                (0..width).map(move |x| {
                    let x = x as f64 - shift_x;
                    let y = y as f64;
                    ((x * 0.19).sin() * 0.31
                        + (y * 0.23).cos() * 0.27
                        + ((x + y) * 0.11).sin() * 0.22
                        + (x * 0.07 + y * 0.17).cos() * 0.2) as f32
                })
            })
            .collect();
        GrayFrame::new(width, height, pixels)
    }
}
