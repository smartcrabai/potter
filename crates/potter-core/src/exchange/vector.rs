use std::collections::{BTreeMap, BTreeSet};

use crate::{
    error::{ErrorCode, PotError, Result},
    eval::{EvaluatedStroke, Snapshot},
    model::SceneDoc,
};

use super::evaluated_meshes;

#[derive(Clone, Copy)]
pub(super) enum Projection {
    Front,
    Back,
    Top,
    Bottom,
    Right,
    Left,
    Isometric,
}

pub(super) struct Bounds {
    pub min: [f64; 2],
    pub max: [f64; 2],
}

impl Bounds {
    pub fn new() -> Self {
        Self {
            min: [f64::INFINITY; 2],
            max: [f64::NEG_INFINITY; 2],
        }
    }

    pub fn include(
        &mut self,
        point: [f64; 2],
        radius: f64,
        non_finite_message: &'static str,
    ) -> Result<()> {
        for (axis, coordinate) in point.into_iter().enumerate() {
            let low = coordinate - radius;
            let high = coordinate + radius;
            if !low.is_finite() || !high.is_finite() {
                return Err(PotError::new(
                    ErrorCode::EvaluationFailed,
                    non_finite_message,
                ));
            }
            self.min[axis] = self.min[axis].min(low);
            self.max[axis] = self.max[axis].max(high);
        }
        Ok(())
    }
}

pub(super) struct ProjectedStroke<'a> {
    pub stroke: &'a EvaluatedStroke,
    pub points: Vec<[f64; 2]>,
    pub max_radius: f64,
}

pub(super) fn project(point: [f64; 3], projection: Projection) -> Result<[f64; 2]> {
    if point.iter().any(|coordinate| !coordinate.is_finite()) {
        return Err(PotError::new(
            ErrorCode::EvaluationFailed,
            "evaluated geometry contains non-finite coordinates",
        ));
    }
    let [x, y, z] = point;
    let projected = match projection {
        Projection::Front => [x, z],
        Projection::Back => [-x, z],
        Projection::Top => [x, y],
        Projection::Bottom => [x, -y],
        Projection::Right => [y, z],
        Projection::Left => [-y, z],
        Projection::Isometric => [
            std::f64::consts::FRAC_1_SQRT_2 * (x - y),
            (x + y) / 6.0_f64.sqrt() + z * (2.0_f64 / 3.0).sqrt(),
        ],
    };
    if projected.iter().any(|coordinate| !coordinate.is_finite()) {
        return Err(PotError::new(
            ErrorCode::EvaluationFailed,
            "projected geometry coordinates are not finite",
        ));
    }
    Ok(projected)
}

pub(super) fn for_each_projected_mesh_edge(
    doc: &SceneDoc,
    snapshot: &Snapshot,
    projection: Projection,
    mut visit: impl FnMut([[f64; 2]; 2]) -> Result<()>,
) -> Result<()> {
    let node_ids_by_name = doc
        .nodes
        .keys()
        .map(|id| (id.as_str(), id))
        .collect::<BTreeMap<_, _>>();
    for mesh in evaluated_meshes(doc, snapshot)? {
        let node_id = node_ids_by_name
            .get(&mesh.id.as_str())
            .copied()
            .ok_or_else(|| {
                PotError::new(
                    ErrorCode::EvaluationFailed,
                    "evaluated mesh node is missing from the scene",
                )
            })?;
        let source = snapshot.meshes.get(node_id).ok_or_else(|| {
            PotError::new(
                ErrorCode::EvaluationFailed,
                "evaluated mesh topology is missing from the snapshot",
            )
        })?;
        let vertex_indices = source
            .vertices
            .iter()
            .enumerate()
            .map(|(index, vertex)| (vertex.id, index))
            .collect::<BTreeMap<_, _>>();
        let mut edges = BTreeSet::new();
        for edge in &source.edges {
            if let (Some(&first), Some(&second)) = (
                vertex_indices.get(&edge.vertices[0]),
                vertex_indices.get(&edge.vertices[1]),
            ) && first != second
            {
                edges.insert(crate::geom::edge_key(first, second));
            }
        }
        for face in &mesh.faces {
            if face.len() < 2 {
                continue;
            }
            for index in 0..face.len() {
                let first = face[index];
                let second = face[(index + 1) % face.len()];
                if first != second {
                    edges.insert(crate::geom::edge_key(first, second));
                }
            }
        }
        for (first, second) in edges {
            let (Some(&point_first), Some(&point_second)) =
                (mesh.positions.get(first), mesh.positions.get(second))
            else {
                continue;
            };
            visit([
                project(point_first, projection)?,
                project(point_second, projection)?,
            ])?;
        }
    }
    Ok(())
}

pub(super) fn for_each_projected_stroke(
    doc: &SceneDoc,
    snapshot: &Snapshot,
    visible_nodes: &BTreeSet<crate::model::Id>,
    projection: Projection,
    mut visit: impl FnMut(ProjectedStroke<'_>) -> Result<()>,
) -> Result<()> {
    for (node_id, node_strokes) in &snapshot.strokes {
        let Some(node) = doc.nodes.get(node_id) else {
            continue;
        };
        if !node.visible || !node.render_visible || !visible_nodes.contains(node_id) {
            continue;
        }
        for stroke in node_strokes {
            if stroke.points_world.is_empty() {
                continue;
            }
            if stroke.color.iter().any(|channel| !channel.is_finite())
                || stroke.radius.iter().any(|radius| !radius.is_finite())
            {
                return Err(PotError::new(
                    ErrorCode::EvaluationFailed,
                    "evaluated Grease Pencil stroke contains non-finite data",
                ));
            }
            let points = stroke
                .points_world
                .iter()
                .map(|point| project(*point, projection))
                .collect::<Result<Vec<_>>>()?;
            let max_radius = stroke
                .radius
                .iter()
                .map(|radius| radius.max(0.0))
                .fold(0.0_f64, f64::max);
            visit(ProjectedStroke {
                stroke,
                points,
                max_radius,
            })?;
        }
    }
    Ok(())
}
