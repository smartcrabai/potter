// SPDX-License-Identifier: GPL-3.0-or-later

use glam::DVec3;

use crate::{
    error::{ErrorCode, PotError, Result},
    geom::clnor::{self, NormalCodeOptions, SetterInput},
};

use super::{
    Mesh, MeshError, mesh_error,
    adaptive_dicing::DicedTopology,
    provenance::{Origin, remap_attributes},
    subdivision_limit::{LimitSurface, Sample},
    subdivision_normals,
    subdivision_uv::{self, FvarLinear, UvLayerValues},
};

/// Builds the render mesh and remaps source attributes through adaptive dicing samples.
pub(super) fn assemble(
    source: &Mesh,
    base_faces: &[Vec<usize>],
    topology: &DicedTopology,
    surface: &LimitSurface,
    uv_layers: &[UvLayerValues],
    linear: FvarLinear,
    preserve_corners: bool,
    input_corner_normals: Option<&[DVec3]>,
) -> Result<Mesh> {
    if base_faces.len() != source.faces.len() {
        return Err(PotError::invalid_argument(
            "adaptive mesh base faces must match the source mesh",
        ));
    }
    if topology.faces.len() != topology.samples.corners.len()
        || topology.positions.len() != topology.samples.vertices.len()
    {
        return Err(invalid_topology(
            "adaptive mesh topology buffers do not match",
        ));
    }
    for face in base_faces {
        if face.len() < 3 || face.iter().any(|&vertex| vertex >= source.vertices.len()) {
            return Err(PotError::invalid_argument(
                "adaptive mesh base face references invalid source geometry",
            ));
        }
    }

    let mut face_corner_starts = Vec::with_capacity(base_faces.len());
    let mut source_corner_count = 0_usize;
    for face in base_faces {
        face_corner_starts.push(source_corner_count);
        source_corner_count = source_corner_count.checked_add(face.len()).ok_or_else(|| {
            PotError::new(ErrorCode::LimitExceeded, "adaptive mesh corner count overflows")
        })?;
    }
    if input_corner_normals.is_some_and(|normals| normals.len() != source_corner_count) {
        return Err(PotError::invalid_argument(
            "adaptive mesh input normals must match source corners",
        ));
    }

    let mut result = Mesh::new();
    let mut origin = Origin {
        vertices: Vec::with_capacity(topology.samples.vertices.len()),
        faces: Vec::with_capacity(topology.faces.len()),
        corners: Some(Vec::new()),
    };
    for (&position, sample) in topology.positions.iter().zip(&topology.samples.vertices) {
        let sample = sample.ok_or_else(|| {
            invalid_topology("adaptive mesh vertex is missing its limit-surface sample")
        })?;
        validate_sample(base_faces, sample)?;
        if !position.is_finite() {
            return Err(PotError::new(
                ErrorCode::EvaluationFailed,
                "adaptive mesh produced a non-finite limit sample",
            ));
        }
        result
            .insert_vertex(position)
            .map_err(|error| mesh_error(&error))?;

        let face = &base_faces[sample.face];
        let mut weights = Vec::with_capacity(face.len());
        add_sample_weights(&mut weights, face.len(), sample, |corner| face[corner]);
        origin.vertices.push(weights);
    }

    let mut output_faces = Vec::with_capacity(topology.faces.len());
    let mut output_corner_count = 0_usize;
    for (indices, samples) in topology.faces.iter().zip(&topology.samples.corners) {
        if indices.len() < 3 || indices.len() != samples.len() {
            return Err(invalid_topology(
                "adaptive mesh face indices and corner samples do not match",
            ));
        }
        let first = *samples
            .first()
            .ok_or_else(|| invalid_topology("adaptive mesh contains an empty face"))?;
        validate_sample(base_faces, first)?;
        let source_face = first.face;
        if samples.iter().any(|sample| sample.face != source_face) {
            return Err(invalid_topology(
                "adaptive mesh face corners span multiple source faces",
            ));
        }
        let material_index = source.faces[source_face].material_index;
        let ids = indices
            .iter()
            .map(|&index| {
                result
                    .vertices
                    .get(index)
                    .map(|vertex| vertex.id)
                    .ok_or(MeshError::InvalidTopology(
                        "adaptive mesh face references a missing vertex",
                    ))
            })
            .collect::<std::result::Result<Vec<_>, _>>()?;
        output_faces.push(Ok((ids, material_index)));
        origin.faces.push(Some(source_face));
        output_corner_count = output_corner_count.checked_add(samples.len()).ok_or_else(|| {
            PotError::new(ErrorCode::LimitExceeded, "adaptive mesh corner count overflows")
        })?;
        let corners = origin.corners.as_mut().expect("corner origins are enabled");
        corners.reserve(samples.len());
        for &sample in samples {
            validate_sample(base_faces, sample)?;
            let face = &base_faces[sample.face];
            let mut weights = Vec::with_capacity(face.len());
            add_sample_weights(&mut weights, face.len(), sample, |corner| {
                face_corner_starts[sample.face] + corner
            });
            corners.push(weights);
        }
    }
    result
        .insert_faces(output_faces)
        .map_err(|error| mesh_error(&error))?;

    remap_attributes(source, &mut result, &origin)?;
    if !uv_layers.is_empty() {
        let loops = uv_layers
            .iter()
            .map(|layer| {
                (
                    layer.name.clone(),
                    surface.face_varying_loops(
                        linear,
                        preserve_corners,
                        &layer.map,
                        &topology.samples,
                    ),
                )
            })
            .collect();
        subdivision_uv::write_loop_uvs(&mut result, loops);
    }

    let normal_options = NormalCodeOptions {
        corner_order: None,
        setter_input: SetterInput::AlreadyNormalized,
    };
    if let Some(input_normals) = input_corner_normals {
        let face_sizes: Vec<usize> = base_faces.iter().map(Vec::len).collect();
        let loop_normals = subdivision_normals::loop_normals(
            input_normals,
            &face_sizes,
            &topology.samples.corners,
        );
        let loop_normals: Vec<Option<DVec3>> = loop_normals.into_iter().map(Some).collect();
        if loop_normals.len() != output_corner_count {
            return Err(invalid_topology(
                "adaptive mesh normal samples do not match output corners",
            ));
        }
        clnor::encode_vectors(&mut result, Some(&loop_normals), normal_options)?;
    } else {
        clnor::write_codes(&mut result, None, normal_options)?;
    }
    super::validated(&result)?;
    Ok(result)
}

fn validate_sample(base_faces: &[Vec<usize>], sample: Sample) -> Result<()> {
    let Some(face) = base_faces.get(sample.face) else {
        return Err(invalid_topology(
            "adaptive mesh sample references a missing source face",
        ));
    };
    let ptex_count = if face.len() == 4 { 1 } else { face.len() };
    if sample.ptex >= ptex_count
        || !sample.u.is_finite()
        || !sample.v.is_finite()
        || !(0.0..=1.0).contains(&sample.u)
        || !(0.0..=1.0).contains(&sample.v)
    {
        return Err(invalid_topology(
            "adaptive mesh sample has invalid ptex coordinates",
        ));
    }
    Ok(())
}

fn add_sample_weights(
    weights: &mut Vec<(usize, f64)>,
    face_size: usize,
    sample: Sample,
    source_index: impl Fn(usize) -> usize,
) {
    let (u, v) = (sample.u, sample.v);
    let bilinear = [
        (1.0 - u) * (1.0 - v),
        u * (1.0 - v),
        u * v,
        (1.0 - u) * v,
    ];
    if face_size == 4 {
        for (corner, weight) in bilinear.into_iter().enumerate() {
            add_weight(weights, source_index(corner), weight);
        }
    } else {
        let corner = sample.ptex;
        let previous = (corner + face_size - 1) % face_size;
        let next = (corner + 1) % face_size;
        let center_weight = bilinear[2] / face_size as f64;
        for index in 0..face_size {
            add_weight(weights, source_index(index), center_weight);
        }
        add_weight(weights, source_index(corner), bilinear[0]);
        add_weight(weights, source_index(corner), bilinear[1] * 0.5);
        add_weight(weights, source_index(next), bilinear[1] * 0.5);
        add_weight(weights, source_index(previous), bilinear[3] * 0.5);
        add_weight(weights, source_index(corner), bilinear[3] * 0.5);
    }
}

fn add_weight(weights: &mut Vec<(usize, f64)>, index: usize, weight: f64) {
    if weight == 0.0 {
        return;
    }
    if let Some((_, current)) = weights.iter_mut().find(|(source, _)| *source == index) {
        *current += weight;
    } else {
        weights.push((index, weight));
    }
}

fn invalid_topology(message: &'static str) -> PotError {
    PotError::new(ErrorCode::InternalError, message)
}
