use glam::DVec3;
use std::collections::BTreeMap;

use crate::model::{EmissionSampling, LightType, Material, SceneDoc, World};

use super::{
    GraphEmissionState, PointLight, graph_emission_uses_auto_conversion,
    material_has_cycles_emission_sampling, shader_graph_emission_estimate,
};
use crate::render::scene::Geometry;

const MAX_LIGHTS_IN_LEAF: usize = 8;
const NUM_BUCKETS: usize = 12;
const PI: f64 = std::f64::consts::PI;

#[derive(Clone, Copy)]
struct Bounds {
    min: DVec3,
    max: DVec3,
    empty: bool,
}

impl Bounds {
    const EMPTY: Self = Self {
        min: DVec3::splat(f64::INFINITY),
        max: DVec3::splat(f64::NEG_INFINITY),
        empty: true,
    };

    fn grow(&mut self, point: DVec3) {
        self.min = self.min.min(point);
        self.max = self.max.max(point);
        self.empty = false;
    }

    fn union(&mut self, other: Self) {
        if !other.empty {
            self.grow(other.min);
            self.grow(other.max);
        }
    }

    fn center(self) -> DVec3 {
        if self.empty {
            DVec3::ZERO
        } else {
            (self.min + self.max) * 0.5
        }
    }

    fn size(self) -> DVec3 {
        if self.empty {
            DVec3::ZERO
        } else {
            self.max - self.min
        }
    }

    fn area(self) -> f64 {
        let size = self.size();
        2.0 * (size.x * size.y + size.y * size.z + size.z * size.x)
    }
}

#[derive(Clone, Copy)]
struct Cone {
    axis: DVec3,
    outer: f64,
    emission: f64,
}

impl Cone {
    const EMPTY: Self = Self {
        axis: DVec3::ZERO,
        outer: f64::MIN_POSITIVE,
        emission: f64::MIN_POSITIVE,
    };

    fn is_empty(self) -> bool {
        self.axis == DVec3::ZERO
    }

    fn merge(a: Self, b: Self) -> Self {
        if a.is_empty() {
            return b;
        }
        if b.is_empty() {
            return a;
        }
        let (a, b) = if b.outer > a.outer { (b, a) } else { (a, b) };
        let cos_distance = a.axis.dot(b.axis).clamp(-1.0, 1.0);
        let distance = cos_distance.acos();
        let emission = a.emission.max(b.emission);
        if a.outer + 5.0e-4 >= (distance + b.outer).min(PI) {
            return Self {
                axis: a.axis,
                outer: a.outer,
                emission,
            };
        }
        let outer = (distance + a.outer + b.outer) * 0.5;
        if outer >= PI {
            return Self {
                axis: a.axis,
                outer: PI,
                emission,
            };
        }
        let axis = if cos_distance < -0.9995 {
            super::make_orthonormals(a.axis).0
        } else {
            let rotation = outer - a.outer;
            let orthogonal = (b.axis - a.axis * cos_distance).normalize_or_zero();
            (a.axis * rotation.cos() + orthogonal * rotation.sin()).normalize_or_zero()
        };
        Self {
            axis,
            outer,
            emission,
        }
    }

    fn measure(self) -> f64 {
        if self.is_empty() {
            return 0.0;
        }
        let theta_w = (self.outer + self.emission).min(PI);
        let cos_outer = self.outer.cos();
        let sin_outer = self.outer.sin();
        2.0 * PI * (1.0 - cos_outer)
            + (PI * 0.5)
                * (2.0 * theta_w * sin_outer
                    - (self.outer - 2.0 * theta_w).cos()
                    - 2.0 * self.outer * sin_outer
                    + cos_outer)
    }
}

#[derive(Clone, Copy)]
struct Measure {
    bounds: Bounds,
    cone: Cone,
    energy: f64,
}

impl Measure {
    const EMPTY: Self = Self {
        bounds: Bounds::EMPTY,
        cone: Cone::EMPTY,
        energy: 0.0,
    };

    fn add(&mut self, other: Self) {
        if other.energy == 0.0 {
            return;
        }
        self.bounds.union(other.bounds);
        self.cone = Cone::merge(self.cone, other.cone);
        self.energy += other.energy;
    }

    fn importance_measure(self) -> f64 {
        if self.energy == 0.0 {
            return 0.0;
        }
        let area = self.bounds.area();
        let area_measure = if area == 0.0 {
            self.bounds.size().length()
        } else {
            area
        };
        self.energy * area_measure * self.cone.measure()
    }
}

#[derive(Clone, Copy)]
enum EmitterKind {
    Lamp(usize),
    World,
    Mesh(usize),
    MeshTriangle,
}

#[derive(Clone, Copy)]
struct Emitter {
    kind: EmitterKind,
    centroid: DVec3,
    importance_centroid: DVec3,
    measure: Measure,
}

struct Node {
    measure: Measure,
    kind: NodeKind,
}

struct MeshSubtree {
    emitters: Vec<Emitter>,
    root: Node,
}

enum NodeKind {
    Inner(Box<Node>, Box<Node>),
    Leaf { emitters: Vec<usize>, distant: bool },
}

#[derive(Clone, Copy)]
struct Importance {
    max: f64,
    min: f64,
}

#[derive(Clone, Copy)]
struct Query {
    point: DVec3,
    normal_or_direction: DVec3,
    distance: f64,
    volume: bool,
    has_transmission: bool,
}
#[derive(Clone, Copy)]
pub(super) struct LightTreeSettings {
    pub(super) normalize: bool,
    pub(super) strength_scale: DVec3,
}

impl Default for LightTreeSettings {
    fn default() -> Self {
        Self {
            normalize: true,
            strength_scale: DVec3::ONE,
        }
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum SampledEmitter {
    Lamp(usize),
    World,
}

pub(super) struct LightTree {
    emitters: Vec<Emitter>,
    mesh_subtrees: Vec<MeshSubtree>,
    lights: Vec<PointLight>,
    root: Option<Node>,
    lamp_emitters: Vec<Option<usize>>,
    world_emitter: Option<usize>,
    paths: Vec<(u64, u32)>,
}

impl LightTree {
    pub(super) fn build(
        doc: &SceneDoc,
        geometry: &Geometry,
        lights: &[PointLight],
        light_tree_settings: &[LightTreeSettings],
        world: Option<&World>,
        world_active: bool,
        world_average_radiance: f64,
    ) -> Self {
        let mut emitters = Vec::new();
        let mut lamp_emitters = vec![None; lights.len()];
        let mut local = Vec::new();
        let mut distant = Vec::new();
        for (index, light) in lights.iter().enumerate() {
            if !eligible_light(light) {
                continue;
            }
            let Some(emitter) = light_emitter(light, index, light_tree_settings[index]) else {
                continue;
            };
            let emitter_index = emitters.len();
            emitters.push(emitter);
            lamp_emitters[index] = Some(emitter_index);
            if light.kind == LightType::Sun {
                distant.push(emitter_index);
            } else {
                local.push(emitter_index);
            }
        }

        let world_emitter = if world_active {
            world.map(|world| {
                let index = emitters.len();
                emitters.push(world_emitter(doc, world, world_average_radiance));
                distant.push(index);
                index
            })
        } else {
            None
        };
        let mut object_bounds = BTreeMap::<u32, Bounds>::new();
        for triangle in &geometry.triangles {
            let bounds = object_bounds
                .entry(triangle.object_index)
                .or_insert(Bounds::EMPTY);
            for position in triangle.positions {
                bounds.grow(position);
            }
        }
        let mut mesh_subtrees = Vec::new();
        for (object_index, triangles) in emissive_meshes(doc, geometry) {
            let centroid = object_bounds
                .get(&object_index)
                .copied()
                .unwrap_or(Bounds::EMPTY)
                .center();
            let subtree = build_mesh_subtree(triangles);
            let index = emitters.len();
            emitters.push(Emitter {
                kind: EmitterKind::Mesh(mesh_subtrees.len()),
                centroid,
                importance_centroid: centroid,
                measure: subtree.root.measure,
            });
            local.push(index);
            mesh_subtrees.push(subtree);
        }
        let local = build_node(&emitters, local, false);
        let distant = build_node(&emitters, distant, true);
        let measure = combine_measure(local.measure, distant.measure);
        let root = Some(Node {
            measure,
            kind: NodeKind::Inner(Box::new(local), Box::new(distant)),
        });
        let mut paths = vec![(0_u64, 0_u32); emitters.len()];
        if let Some(root) = &root {
            collect_paths(root, 0, 0, &mut paths);
        }
        Self {
            emitters,
            mesh_subtrees,
            lights: lights.to_vec(),
            root,
            lamp_emitters,
            world_emitter,
            paths,
        }
    }

    pub(super) fn lamp_pdf(
        &self,
        index: usize,
        point: DVec3,
        normal_or_direction: DVec3,
        distance: f64,
        volume: bool,
        has_transmission: bool,
    ) -> f64 {
        let Some(emitter) = self.lamp_emitters.get(index).copied().flatten() else {
            return 0.0;
        };
        self.pdf(
            emitter,
            Query {
                point,
                normal_or_direction,
                distance,
                volume,
                has_transmission,
            },
        )
    }

    pub(super) fn world_pdf(
        &self,
        point: DVec3,
        normal_or_direction: DVec3,
        distance: f64,
        volume: bool,
        has_transmission: bool,
    ) -> f64 {
        self.world_emitter.map_or(0.0, |emitter| {
            self.pdf(
                emitter,
                Query {
                    point,
                    normal_or_direction,
                    distance,
                    volume,
                    has_transmission,
                },
            )
        })
    }

    pub(super) fn lamp_pdfs(
        &self,
        point: DVec3,
        normal_or_direction: DVec3,
        distance: f64,
        volume: bool,
        has_transmission: bool,
        output: &mut Vec<f64>,
    ) {
        output.resize(self.lamp_emitters.len(), 0.0);
        for (index, emitter) in self.lamp_emitters.iter().enumerate() {
            if let Some(emitter) = emitter {
                output[index] = self.pdf(
                    *emitter,
                    Query {
                        point,
                        normal_or_direction,
                        distance,
                        volume,
                        has_transmission,
                    },
                );
            }
        }
    }

    /// Traverses the same max/min midpoint CDF as `pdf`, remapping each selected interval.
    pub(super) fn sample(
        &self,
        sample: f64,
        point: DVec3,
        normal_or_direction: DVec3,
        distance: f64,
        volume: bool,
        has_transmission: bool,
    ) -> Option<(SampledEmitter, f64)> {
        if !(0.0..1.0).contains(&sample) {
            return None;
        }
        let query = Query {
            point,
            normal_or_direction,
            distance,
            volume,
            has_transmission,
        };
        let mut remaining = sample;
        let mut probability = 1.0;
        let mut node = self.root.as_ref()?;
        loop {
            match &node.kind {
                NodeKind::Inner(left, right) => {
                    let left_probability = self.left_probability(left, right, query)?;
                    if remaining < left_probability {
                        remaining /= left_probability;
                        probability *= left_probability;
                        node = left;
                    } else {
                        let right_probability = 1.0 - left_probability;
                        if right_probability <= 0.0 {
                            return None;
                        }
                        remaining = (remaining - left_probability) / right_probability;
                        probability *= right_probability;
                        node = right;
                    }
                }
                NodeKind::Leaf { emitters, .. } => {
                    for emitter in emitters {
                        let emitter_probability = self.leaf_probability(emitters, *emitter, query);
                        if emitter_probability <= 0.0 {
                            continue;
                        }
                        if remaining < emitter_probability {
                            match self.emitters[*emitter].kind {
                                EmitterKind::Lamp(index) => {
                                    return Some((
                                        SampledEmitter::Lamp(index),
                                        probability * emitter_probability,
                                    ));
                                }
                                EmitterKind::World => {
                                    return Some((
                                        SampledEmitter::World,
                                        probability * emitter_probability,
                                    ));
                                }
                                EmitterKind::Mesh(_) | EmitterKind::MeshTriangle => return None,
                            }
                        }
                        remaining -= emitter_probability;
                    }
                    return None;
                }
            }
        }
    }

    fn left_probability(&self, left: &Node, right: &Node, query: Query) -> Option<f64> {
        let left_importance = child_importance(
            left,
            &self.emitters,
            &self.lights,
            &self.mesh_subtrees,
            query,
        );
        let right_importance = child_importance(
            right,
            &self.emitters,
            &self.lights,
            &self.mesh_subtrees,
            query,
        );
        let total_max = left_importance.max + right_importance.max;
        if total_max <= 0.0 {
            return None;
        }
        let total_min = left_importance.min + right_importance.min;
        let probability_max = left_importance.max / total_max;
        let probability_min = if total_min > 0.0 {
            left_importance.min / total_min
        } else {
            (if left_importance.max > 0.0 { 0.5 } else { 0.0 })
                + (if right_importance.max == 0.0 {
                    0.5
                } else {
                    0.0
                })
        };
        Some(f64::midpoint(probability_max, probability_min))
    }

    fn leaf_probability(&self, emitters: &[usize], target: usize, query: Query) -> f64 {
        let mut total_max = 0.0;
        let mut total_min = 0.0;
        let mut positive_max = 0_usize;
        let mut target_importance = Importance { max: 0.0, min: 0.0 };
        for emitter in emitters {
            let current = emitter_importance(
                self.emitters[*emitter],
                &self.lights,
                &self.mesh_subtrees,
                query,
            );
            total_max += current.max;
            total_min += current.min;
            positive_max += usize::from(current.max > 0.0);
            if *emitter == target {
                target_importance = current;
            }
        }
        if target_importance.max <= 0.0 || total_max <= 0.0 {
            return 0.0;
        }
        let minimum_probability = if total_min > 0.0 {
            target_importance.min / total_min
        } else if positive_max > 0 {
            1.0 / positive_max as f64
        } else {
            0.0
        };
        f64::midpoint(target_importance.max / total_max, minimum_probability)
    }

    fn pdf(&self, target: usize, query: Query) -> f64 {
        let Some(root) = &self.root else {
            return 0.0;
        };
        let Some(&(path_bits, path_length)) = self.paths.get(target) else {
            return 0.0;
        };
        let mut node = root;
        let mut pdf = 1.0;
        for depth in 0..path_length {
            let NodeKind::Inner(left, right) = &node.kind else {
                break;
            };
            let Some(left_probability) = self.left_probability(left, right, query) else {
                return 0.0;
            };
            let go_right = path_bits & (1_u64 << depth) != 0;
            if go_right {
                pdf *= 1.0 - left_probability;
                node = right;
            } else {
                pdf *= left_probability;
                node = left;
            }
        }
        let NodeKind::Leaf { emitters, .. } = &node.kind else {
            return 0.0;
        };
        pdf * self.leaf_probability(emitters, target, query)
    }
}

struct TriangleMeasure {
    centroid: DVec3,
    measure: Measure,
}

fn build_mesh_subtree(triangles: Vec<TriangleMeasure>) -> MeshSubtree {
    let emitters: Vec<Emitter> = triangles
        .into_iter()
        .map(|triangle| Emitter {
            kind: EmitterKind::MeshTriangle,
            centroid: triangle.centroid,
            importance_centroid: triangle.centroid,
            measure: triangle.measure,
        })
        .collect();
    debug_assert!(!emitters.is_empty());
    let indices = (0..emitters.len()).collect();
    let root = build_node(&emitters, indices, false);
    MeshSubtree { emitters, root }
}

fn eligible_light(light: &PointLight) -> bool {
    !(light.kind == LightType::Area && light.is_portal)
        && light.energy > 0.0
        && light.color.max_element() > 0.0
        && (light.kind != LightType::Area
            || light.area.is_some_and(|area| {
                area.area > 0.0 && area.axis_u != DVec3::ZERO && area.axis_v != DVec3::ZERO
            }))
}

fn light_emitter(light: &PointLight, index: usize, settings: LightTreeSettings) -> Option<Emitter> {
    let mut bounds = Bounds::EMPTY;
    let mut cone = Cone {
        axis: light.direction.normalize_or_zero(),
        outer: 0.0,
        emission: 0.0,
    };
    let mut strength = light.color * light.energy * settings.strength_scale;
    if !settings.normalize {
        strength *= light_area(light)?;
    }

    match light.kind {
        LightType::Point => {
            cone.outer = PI;
            cone.emission = PI * 0.5;
            let radius = DVec3::splat(light.radius);
            bounds.grow(light.position - radius);
            bounds.grow(light.position + radius);
            strength *= 0.25 / PI;
        }
        LightType::Spot => {
            let transform = light.world_transform;
            let len_u = transform.x_axis.truncate().length();
            let len_v = transform.y_axis.truncate().length();
            let len_w = transform.z_axis.truncate().length();
            let mut emission = (light.spot_size * 0.5).min(PI * 0.5);
            if (PI * 0.5 - emission).abs() >= 1.0e-6 {
                emission = (emission.tan() * len_u.max(len_v) / len_w).atan().abs();
            }
            cone.emission = emission;
            let radius = DVec3::splat(light.radius);
            bounds.grow(light.position - radius);
            bounds.grow(light.position + radius);
            strength *= 0.25 / PI;
        }
        LightType::Area => {
            let shape = light.area?;
            cone.emission = light
                .spread
                .map_or(PI * 0.5, crate::render::light_model::Spread::half);
            let extent_u = shape.axis_u * shape.half_u;
            let extent_v = shape.axis_v * shape.half_v;
            for sign_u in [-1.0, 1.0] {
                for sign_v in [-1.0, 1.0] {
                    bounds.grow(light.position + sign_u * extent_u + sign_v * extent_v);
                }
            }
            strength /= PI;
        }
        LightType::Sun => {
            cone.emission = light.spot_size * 0.5;
        }
    }
    let importance_centroid = if light.kind == LightType::Sun {
        light.direction
    } else {
        light.position
    };
    Some(Emitter {
        kind: EmitterKind::Lamp(index),
        centroid: light.position,
        importance_centroid,
        measure: Measure {
            bounds,
            cone,
            energy: average_abs(strength),
        },
    })
}
fn light_area(light: &PointLight) -> Option<f64> {
    match light.kind {
        LightType::Point | LightType::Spot => Some(if light.radius == 0.0 {
            4.0
        } else {
            4.0 * PI * light.radius * light.radius
        }),
        LightType::Area => light.area.map(|area| area.area),
        LightType::Sun => Some(if light.spot_size == 0.0 {
            1.0
        } else {
            PI * (light.spot_size * 0.5).sin().powi(2)
        }),
    }
}

fn world_emitter(doc: &SceneDoc, world: &World, average_radiance: f64) -> Emitter {
    let estimate = if let Some(graph_id) = &world.node_tree {
        shader_graph_emission_estimate(
            doc,
            graph_id,
            "ShaderNodeOutputWorld",
            "Surface",
            &mut GraphEmissionState::default(),
        )
    } else {
        DVec3::from_array(world.color) * world.strength
    };
    Emitter {
        kind: EmitterKind::World,
        centroid: DVec3::Z,
        importance_centroid: DVec3::Z,
        measure: Measure {
            bounds: Bounds::EMPTY,
            cone: Cone {
                axis: DVec3::Z,
                outer: PI,
                emission: 0.0,
            },
            energy: average_radiance * average_abs(estimate) * PI,
        },
    }
}

fn emissive_triangle_cone(
    sampling: EmissionSampling,
    positions: [DVec3; 3],
    normal: DVec3,
) -> Cone {
    let (axis, outer) = match sampling {
        EmissionSampling::Back => (-normal, 0.0),
        EmissionSampling::Front => (normal, 0.0),
        EmissionSampling::Auto | EmissionSampling::FrontBack | EmissionSampling::None => {
            ((positions[0] - positions[1]).normalize_or_zero(), PI * 0.5)
        }
    };
    Cone {
        axis,
        outer,
        emission: PI * 0.5,
    }
}

fn emissive_meshes(doc: &SceneDoc, geometry: &Geometry) -> Vec<(u32, Vec<TriangleMeasure>)> {
    let material_estimates: Vec<Option<DVec3>> = geometry
        .materials
        .iter()
        .map(|material| {
            material_has_cycles_emission_sampling(doc, material)
                .then(|| material_emission_estimate(doc, material))
        })
        .collect();
    let mut meshes = BTreeMap::<u32, Vec<TriangleMeasure>>::new();
    for triangle in &geometry.triangles {
        if triangle.ray_mask
            & (crate::model::RayKind::Diffuse.bit()
                | crate::model::RayKind::Glossy.bit()
                | crate::model::RayKind::Transmission.bit()
                | crate::model::RayKind::VolumeScatter.bit())
            == 0
            || material_estimates
                .get(triangle.material_index)
                .copied()
                .flatten()
                .is_none()
        {
            continue;
        }
        let Some(material) = geometry.materials.get(triangle.material_index) else {
            continue;
        };
        let area = triangle_area(triangle.positions);
        if area <= 0.0 || !area.is_finite() {
            continue;
        }
        let Some(estimate) = material_estimates[triangle.material_index] else {
            continue;
        };
        let normal = (triangle.positions[1] - triangle.positions[0])
            .cross(triangle.positions[2] - triangle.positions[0])
            .normalize_or_zero();
        let cone = emissive_triangle_cone(material.emission_sampling, triangle.positions, normal);

        let mut bounds = Bounds::EMPTY;
        for point in triangle.positions {
            bounds.grow(point);
        }
        let centroid =
            (triangle.positions[0] + triangle.positions[1] + triangle.positions[2]) / 3.0;
        meshes
            .entry(triangle.object_index)
            .or_default()
            .push(TriangleMeasure {
                centroid,
                measure: Measure {
                    bounds,
                    cone,

                    energy: area * average_abs(estimate),
                },
            });
    }
    meshes.into_iter().collect()
}

fn material_emission_estimate(doc: &SceneDoc, material: &Material) -> DVec3 {
    let Some(graph_id) = &material.node_tree else {
        return DVec3::from_array(material.emission_color) * material.emission_strength;
    };
    let estimate = shader_graph_emission_estimate(
        doc,
        graph_id,
        "ShaderNodeOutputMaterial",
        "Surface",
        &mut GraphEmissionState::default(),
    );
    if graph_emission_uses_auto_conversion(doc, graph_id) {
        estimate * 0.1
    } else {
        estimate
    }
}

fn triangle_area(points: [DVec3; 3]) -> f64 {
    (points[1] - points[0])
        .cross(points[2] - points[0])
        .length()
        * 0.5
}

fn average_abs(color: DVec3) -> f64 {
    (color.x.abs() + color.y.abs() + color.z.abs()) / 3.0
}

fn combine_measure(a: Measure, b: Measure) -> Measure {
    let mut result = a;
    result.add(b);
    result
}

fn build_node(emitters: &[Emitter], mut indices: Vec<usize>, distant: bool) -> Node {
    if distant {
        let measure = aggregate_measure(emitters, &indices);
        return Node {
            measure,
            kind: NodeKind::Leaf {
                emitters: indices,
                distant: true,
            },
        };
    }
    let (measure, middle, split_dim, should_split) = split_plan(emitters, &indices);
    if !should_split {
        return Node {
            measure,
            kind: NodeKind::Leaf {
                emitters: indices,
                distant: false,
            },
        };
    }
    if split_dim >= 0 {
        indices.sort_by(|left, right| {
            emitters[*left].centroid[split_dim as usize]
                .total_cmp(&emitters[*right].centroid[split_dim as usize])
        });
    }
    let middle = middle.clamp(1, indices.len().saturating_sub(1));
    let right_indices = indices.split_off(middle);
    let left = build_node(emitters, indices, false);
    let right = build_node(emitters, right_indices, false);
    Node {
        measure,
        kind: NodeKind::Inner(Box::new(left), Box::new(right)),
    }
}

fn aggregate_measure(emitters: &[Emitter], indices: &[usize]) -> Measure {
    indices.iter().fold(Measure::EMPTY, |mut total, index| {
        total.add(emitters[*index].measure);
        total
    })
}

fn split_plan(emitters: &[Emitter], indices: &[usize]) -> (Measure, usize, i32, bool) {
    if indices.len() < 2 {
        let measure = indices
            .first()
            .map_or(Measure::EMPTY, |index| emitters[*index].measure);
        return (measure, 0, -1, false);
    }
    let mut centroid_bounds = Bounds::EMPTY;
    for index in indices {
        centroid_bounds.grow(emitters[*index].centroid);
    }
    let extent = centroid_bounds.size();
    let max_extent = extent.max_element().max(0.0);
    let mut total_cost = 0.0;
    let mut min_cost = f64::MAX;
    let mut middle = indices.len() / 2;
    let mut split_dim = -1;
    let mut node_measure = Measure::EMPTY;
    for dim in 0..3 {
        let dimension_extent = extent[dim];
        let mut buckets = [Measure::EMPTY; NUM_BUCKETS];
        let mut counts = [0_usize; NUM_BUCKETS];
        let inverse_extent;
        if dimension_extent == 0.0 {
            if dim != 0 {
                continue;
            }
            inverse_extent = f64::MAX;
            for index in indices {
                buckets[0].add(emitters[*index].measure);
                counts[0] += 1;
            }
        } else {
            inverse_extent = 1.0 / dimension_extent;
            for index in indices {
                let normalized =
                    (emitters[*index].centroid[dim] - centroid_bounds.min[dim]) * inverse_extent;
                let bucket = ((NUM_BUCKETS as f64 * normalized) as usize).min(NUM_BUCKETS - 1);
                buckets[bucket].add(emitters[*index].measure);
                counts[bucket] += 1;
            }
        }
        let mut left_measures = [Measure::EMPTY; NUM_BUCKETS - 1];
        let mut left_counts = [0_usize; NUM_BUCKETS - 1];
        left_measures[0] = buckets[0];
        left_counts[0] = counts[0];
        for bucket in 1..NUM_BUCKETS - 1 {
            left_measures[bucket] = combine_measure(left_measures[bucket - 1], buckets[bucket]);
            left_counts[bucket] = left_counts[bucket - 1] + counts[bucket];
        }
        if dim == 0 {
            node_measure =
                combine_measure(left_measures[NUM_BUCKETS - 2], buckets[NUM_BUCKETS - 1]);
            if extent == DVec3::ZERO {
                break;
            }
            if dimension_extent == 0.0 {
                continue;
            }
            total_cost = node_measure.importance_measure();
            if total_cost == 0.0 {
                break;
            }
        }
        let mut right_measures = [Measure::EMPTY; NUM_BUCKETS - 1];
        right_measures[NUM_BUCKETS - 2] = buckets[NUM_BUCKETS - 1];
        for bucket in (0..NUM_BUCKETS - 2).rev() {
            right_measures[bucket] =
                combine_measure(right_measures[bucket + 1], buckets[bucket + 1]);
        }
        let regularization = max_extent * inverse_extent;
        for split in 0..NUM_BUCKETS - 1 {
            let cost = regularization
                * (left_measures[split].importance_measure()
                    + right_measures[split].importance_measure());
            if cost < total_cost && cost < min_cost {
                min_cost = cost;
                split_dim = dim as i32;
                middle = left_counts[split];
            }
        }
    }
    (
        node_measure,
        middle,
        split_dim,
        min_cost < total_cost || indices.len() > MAX_LIGHTS_IN_LEAF,
    )
}

fn collect_paths(node: &Node, prefix: u64, depth: u32, paths: &mut [(u64, u32)]) {
    match &node.kind {
        NodeKind::Leaf { emitters, .. } => {
            for emitter in emitters {
                paths[*emitter] = (prefix, depth);
            }
        }
        NodeKind::Inner(left, right) => {
            collect_paths(left, prefix, depth + 1, paths);
            collect_paths(right, prefix | (1_u64 << depth), depth + 1, paths);
        }
    }
}

fn child_importance(
    node: &Node,
    emitters: &[Emitter],
    lights: &[PointLight],
    mesh_subtrees: &[MeshSubtree],
    query: Query,
) -> Importance {
    match &node.kind {
        NodeKind::Leaf {
            emitters: items, ..
        } if items.len() == 1 => {
            emitter_importance(emitters[items[0]], lights, mesh_subtrees, query)
        }
        NodeKind::Leaf {
            emitters: items, ..
        } if items.is_empty() => Importance { max: 0.0, min: 0.0 },
        NodeKind::Leaf { distant, .. } => node_importance(node.measure, query, *distant),
        NodeKind::Inner(_, _) => node_importance(node.measure, query, false),
    }
}

#[derive(Clone, Copy)]
enum TreeLightKind {
    Sun,
    Point {
        radius: f64,
        sphere: bool,
    },
    Spot {
        radius: f64,
        sphere: bool,
        blend: f64,
    },
    Area {
        shape: Option<crate::render::light_model::AreaShape>,
    },
}

fn tree_light_kind(light: &PointLight) -> TreeLightKind {
    match light.kind {
        LightType::Sun => TreeLightKind::Sun,
        LightType::Point => TreeLightKind::Point {
            radius: light.radius,
            sphere: !light.soft_falloff && light.radius != 0.0,
        },
        LightType::Spot => TreeLightKind::Spot {
            radius: light.radius,
            sphere: !light.soft_falloff && light.radius != 0.0,
            blend: light.spot_blend,
        },
        LightType::Area => TreeLightKind::Area { shape: light.area },
    }
}

fn emitter_importance(
    emitter: Emitter,
    lights: &[PointLight],
    mesh_subtrees: &[MeshSubtree],
    query: Query,
) -> Importance {
    match emitter.kind {
        EmitterKind::Mesh(index) => {
            let subtree = &mesh_subtrees[index];
            debug_assert!(!subtree.emitters.is_empty());
            return node_importance(subtree.root.measure, query, false);
        }
        EmitterKind::MeshTriangle => return node_importance(emitter.measure, query, false),
        EmitterKind::World | EmitterKind::Lamp(_) => {}
    }

    let mut cone = emitter.measure.cone;
    let mut centroid = emitter.importance_centroid;
    let mut theta_d = 1.0;
    let mut distance: [f64; 2];
    let mut point_to_centroid;
    let mut cos_theta_u;
    let mut energy = emitter.measure.energy;
    let mut point_at_segment = query.point;
    if query.volume {
        let direction = query.normal_or_direction.normalize_or_zero();
        let center_to_point = centroid - query.point;
        let closest_t = center_to_point.dot(direction);
        point_at_segment += direction * closest_t.clamp(0.0, query.distance);
        let perpendicular = (center_to_point - direction * closest_t).length();
        theta_d = if query.distance.is_infinite() {
            closest_t.atan2(perpendicular) + PI * 0.5
        } else {
            let denominator =
                perpendicular - closest_t * safe_divide(query.distance - closest_t, perpendicular);
            query.distance.atan2(denominator)
        };
    }
    match emitter.kind {
        EmitterKind::World => {
            if query.volume {
                theta_d = if query.distance.is_infinite() {
                    1.0
                } else {
                    query.distance
                };
            }
            centroid = DVec3::Z;
            cone.axis = -DVec3::Z;
            point_to_centroid = -centroid;
            cos_theta_u = -1.0;
            distance = [1.0, 1.0];
        }
        EmitterKind::Lamp(index) => {
            let light = &lights[index];
            let kind = tree_light_kind(light);
            if matches!(kind, TreeLightKind::Point { .. }) {
                cone.axis = (query.point - centroid).normalize_or_zero();
                cone.outer = 0.0;
            }
            let mut min_distance = 0.0;
            point_to_centroid = safe_normalize_len(centroid - point_at_segment, &mut min_distance);
            distance = [min_distance, min_distance];
            match kind {
                TreeLightKind::Sun => {
                    if query.volume {
                        theta_d = if query.distance.is_infinite() {
                            1.0
                        } else {
                            query.distance
                        };
                    }
                    cos_theta_u = cone.emission.cos();
                    distance = [1.0 / cos_theta_u, 1.0];
                    point_to_centroid = -centroid.normalize_or_zero();
                }
                TreeLightKind::Point { radius, sphere } => {
                    if query.volume {
                        cos_theta_u = 1.0;
                    } else if sphere {
                        if min_distance > radius {
                            cos_theta_u = cos_from_sin(radius / min_distance);
                            distance[0] = min_distance / cos_theta_u;
                        } else {
                            cos_theta_u = -1.0;
                            distance = [
                                radius / std::f64::consts::SQRT_2,
                                radius / std::f64::consts::SQRT_2,
                            ];
                        }
                    } else {
                        let hypotenuse = radius.hypot(min_distance);
                        cos_theta_u = safe_divide(min_distance, hypotenuse);
                        distance[0] = hypotenuse;
                    }
                }
                TreeLightKind::Spot {
                    radius,
                    sphere,
                    blend,
                } => {
                    cos_theta_u = if sphere && min_distance > radius {
                        cos_from_sin(radius / min_distance)
                    } else if sphere {
                        -1.0
                    } else {
                        safe_divide(min_distance, radius.hypot(min_distance))
                    };
                    if !query.volume {
                        distance = if sphere {
                            if min_distance > radius {
                                [min_distance / cos_theta_u, min_distance]
                            } else {
                                [
                                    radius / std::f64::consts::SQRT_2,
                                    radius / std::f64::consts::SQRT_2,
                                ]
                            }
                        } else {
                            [radius.hypot(min_distance), min_distance]
                        };
                        let cos_min_outgoing =
                            (cone.axis.dot(-point_to_centroid).clamp(-1.0, 1.0).acos()
                                - cos_theta_u.clamp(-1.0, 1.0).acos())
                            .max(0.0)
                            .cos();
                        let cos_half_spot = (light.spot_size * 0.5).cos();
                        let spot_smooth = if blend == 0.0 {
                            f64::MAX
                        } else {
                            1.0 / ((1.0 - cos_half_spot) * blend)
                        };
                        energy *= smoothstep((cos_min_outgoing - cos_half_spot) * spot_smooth);
                    }
                }
                TreeLightKind::Area { shape } => {
                    let Some(shape) = shape else {
                        return Importance { max: 0.0, min: 0.0 };
                    };
                    cos_theta_u = f64::MAX;
                    let extent_u = shape.axis_u * (2.0 * shape.half_u);
                    let extent_v = shape.axis_v * (2.0 * shape.half_v);
                    let shape_above_surface =
                        query.normal_or_direction.dot(centroid - point_at_segment)
                            + query.normal_or_direction.dot(extent_u).abs()
                            + query.normal_or_direction.dot(extent_v).abs()
                            > 0.0;
                    let mut max_distance = min_distance;
                    for corner in 0..4 {
                        let point = centroid
                            + if corner & 1 == 0 { -0.5 } else { 0.5 } * extent_u
                            + if corner & 2 == 0 { -0.5 } else { 0.5 } * extent_v;
                        let mut corner_distance = 0.0;
                        let point_to_corner =
                            safe_normalize_len(point - point_at_segment, &mut corner_distance);
                        cos_theta_u = cos_theta_u.min(point_to_centroid.dot(point_to_corner));
                        if !query.volume {
                            max_distance = max_distance.max(corner_distance);
                        }
                    }
                    distance = [
                        if query.volume {
                            min_distance
                        } else {
                            max_distance
                        },
                        min_distance,
                    ];
                    cos_theta_u = cos_theta_u.min(1.0);
                    let front_facing = cone.axis.dot(point_to_centroid) < 0.0;
                    if !query.has_transmission && (!front_facing || !shape_above_surface) {
                        return Importance { max: 0.0, min: 0.0 };
                    }
                }
            }
        }
        EmitterKind::Mesh(_) | EmitterKind::MeshTriangle => unreachable!(),
    }
    let sun = matches!(
        emitter.kind,
        EmitterKind::Lamp(index) if matches!(tree_light_kind(&lights[index]), TreeLightKind::Sun)
    );
    if query.volume && sun {
        point_to_centroid = -cone.axis;
    } else if query.volume {
        point_to_centroid = -compute_v(
            centroid,
            query.point,
            query.normal_or_direction,
            cone.axis,
            query.distance,
        );
    }
    importance(
        query,
        point_to_centroid,
        cos_theta_u,
        cone,
        distance[0],
        distance[1],
        energy,
        theta_d,
    )
}

fn node_importance(measure: Measure, query: Query, distant: bool) -> Importance {
    if measure.energy == 0.0 {
        return Importance { max: 0.0, min: 0.0 };
    }
    let cone = measure.cone;
    if distant {
        let point_to_centroid = -cone.axis;
        let cos_theta_u = (cone.outer + cone.emission).cos();
        let theta_d = if query.volume {
            if query.distance.is_infinite() {
                1.0
            } else {
                query.distance
            }
        } else {
            1.0
        };
        return importance(
            query,
            point_to_centroid,
            cos_theta_u,
            cone,
            1.0,
            1.0,
            measure.energy,
            theta_d,
        );
    }
    let mut theta_d = 1.0;
    let centroid = measure.bounds.center();
    let distance;
    let point_to_centroid;
    let cos_theta_u;
    if query.volume {
        let direction = query.normal_or_direction.normalize_or_zero();
        let closest_t = (centroid - query.point).dot(direction);
        let closest_point = query.point + direction * closest_t.clamp(0.0, query.distance);
        distance = (centroid - query.point - direction * closest_t).length();
        theta_d = if query.distance.is_infinite() {
            closest_t.atan2(distance) + PI * 0.5
        } else {
            let denominator =
                distance - closest_t * safe_divide(query.distance - closest_t, distance);
            query.distance.atan2(denominator)
        };
        point_to_centroid = -compute_v(centroid, query.point, direction, cone.axis, query.distance);
        cos_theta_u = cos_bound_subtended_angle(measure.bounds, centroid, closest_point);
    } else {
        let extent = measure.bounds.max - centroid;
        let visible = query.has_transmission
            || (query.normal_or_direction.dot(centroid - query.point)
                + query.normal_or_direction.abs().dot(extent.abs())
                > 0.0);
        if !visible {
            return Importance { max: 0.0, min: 0.0 };
        }
        let mut min_distance = 0.0;
        point_to_centroid = safe_normalize_len(centroid - query.point, &mut min_distance);
        distance = min_distance;
        cos_theta_u = cos_bound_subtended_angle(measure.bounds, centroid, query.point);
    }
    let distance = distance.max(0.5 * (centroid - measure.bounds.max).length());
    importance(
        query,
        point_to_centroid,
        cos_theta_u,
        cone,
        distance,
        distance,
        measure.energy,
        theta_d,
    )
}

fn importance(
    query: Query,
    point_to_centroid: DVec3,
    cos_theta_u: f64,
    cone: Cone,
    max_distance: f64,
    min_distance: f64,
    energy: f64,
    theta_d: f64,
) -> Importance {
    if energy == 0.0 || min_distance <= 0.0 {
        return Importance { max: 0.0, min: 0.0 };
    }
    let sin_theta_u = sin_from_cos(cos_theta_u);
    let mut cos_min_incidence = 1.0;
    let mut cos_max_incidence = 1.0;
    if !query.volume {
        let cos_theta_i = if query.has_transmission {
            point_to_centroid.dot(query.normal_or_direction).abs()
        } else {
            point_to_centroid.dot(query.normal_or_direction)
        };
        let sin_theta_i = sin_from_cos(cos_theta_i);
        cos_min_incidence = if cos_theta_i >= cos_theta_u {
            1.0
        } else {
            cos_theta_i * cos_theta_u + sin_theta_i * sin_theta_u
        };
        if !query.has_transmission && cos_min_incidence < 0.0 {
            return Importance { max: 0.0, min: 0.0 };
        }
        cos_max_incidence = (cos_theta_i * cos_theta_u - sin_theta_i * sin_theta_u).max(0.0);
    }
    let cos_theta = cone.axis.dot(-point_to_centroid).clamp(-1.0, 1.0);
    let sin_theta = sin_from_cos(cos_theta);
    let cos_theta_minus_u = cos_theta * cos_theta_u + sin_theta * sin_theta_u;
    let cos_outer = cone.outer.cos();
    let sin_outer = cone.outer.sin();
    let cos_min_outgoing = if cos_theta >= cos_theta_u || cos_theta_minus_u >= cos_outer {
        1.0
    } else if cone.outer + cone.emission > PI
        || cos_theta_minus_u > (cone.outer + cone.emission).cos()
    {
        cos_theta_minus_u * cos_outer + sin_from_cos(cos_theta_minus_u) * sin_outer
    } else {
        return Importance { max: 0.0, min: 0.0 };
    };
    let max = (cos_min_incidence
        * energy
        * cos_min_outgoing
        * if query.volume {
            theta_d / min_distance
        } else {
            1.0 / min_distance.powi(2)
        })
    .abs();
    if query.volume {
        return Importance { max, min: 0.0 };
    }
    let cos_theta_plus_u = cos_theta * cos_theta_u - sin_theta * sin_theta_u;
    let min = if cone.emission - cone.outer < 0.0
        || cos_theta < 0.0
        || cos_theta_u < 0.0
        || cos_theta_plus_u < (cone.emission - cone.outer).cos()
    {
        0.0
    } else {
        let cos_max_outgoing =
            cos_theta_plus_u * cos_outer - sin_from_cos(cos_theta_plus_u) * sin_outer;
        (cos_max_incidence * energy * cos_max_outgoing / max_distance.powi(2)).abs()
    };
    Importance { max, min }
}

fn compute_v(centroid: DVec3, point: DVec3, direction: DVec3, axis: DVec3, t: f64) -> DVec3 {
    let unnormalized_v0 = point - centroid;
    let unnormalized_v1 = unnormalized_v0 + direction * t.min(1.0e12);
    let v0 = unnormalized_v0.normalize_or_zero();
    let v1 = unnormalized_v1.normalize_or_zero();
    let o0 = v0;
    let toward = v1 - o0 * o0.dot(v1);
    let o1 = toward.normalize_or_zero();
    let dot_o0_a = o0.dot(axis);
    let dot_o1_a = o1.dot(axis);
    let inverse_length = (dot_o0_a.powi(2) + dot_o1_a.powi(2)).sqrt().recip();
    let cos_phi0 = dot_o0_a * inverse_length;
    if dot_o1_a < 0.0 || v0.dot(v1) > cos_phi0 {
        if dot_o0_a > v1.dot(axis) { v0 } else { v1 }
    } else {
        cos_phi0 * o0 + dot_o1_a * inverse_length * o1
    }
}

fn cos_bound_subtended_angle(bounds: Bounds, centroid: DVec3, point: DVec3) -> f64 {
    let distance_squared = (point - centroid).length_squared();
    let radius_squared = (bounds.max - centroid).length_squared();
    if distance_squared <= radius_squared {
        -1.0
    } else {
        (1.0 - radius_squared / distance_squared).max(0.0).sqrt()
    }
}

fn safe_normalize_len(vector: DVec3, length: &mut f64) -> DVec3 {
    *length = vector.length();
    if *length > 0.0 {
        vector / *length
    } else {
        DVec3::ZERO
    }
}

fn safe_divide(numerator: f64, denominator: f64) -> f64 {
    if denominator == 0.0 {
        0.0
    } else {
        numerator / denominator
    }
}

fn cos_from_sin(sine: f64) -> f64 {
    (1.0 - sine * sine).max(0.0).sqrt()
}

fn sin_from_cos(cosine: f64) -> f64 {
    (1.0 - cosine * cosine).max(0.0).sqrt()
}

fn smoothstep(value: f64) -> f64 {
    let value = value.clamp(0.0, 1.0);
    value * value * (3.0 - 2.0 * value)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::RayKind;
    use glam::DMat4;

    fn point_light(position: DVec3, energy: f64) -> PointLight {
        PointLight {
            position,
            direction: -DVec3::Z,
            world_transform: DMat4::IDENTITY,
            kind: LightType::Point,
            color: DVec3::ONE,
            energy,
            radius: 0.0,
            soft_falloff: true,
            spot_size: 0.0,
            spot_blend: 0.0,
            area: None,
            spread: None,
            use_mis: true,
            is_portal: false,
            max_bounces: 1024,
            volume_factor: 1.0,
            use_shadow: Some(true),
            volume_radius_squared: 0.0,
            ray_mask: RayKind::Diffuse.bit(),
            camera_shape_eligible: false,
        }
    }

    fn point_tree(lights: Vec<PointLight>) -> LightTree {
        let mut emitters = Vec::with_capacity(lights.len());
        let mut indices = Vec::with_capacity(lights.len());
        for (index, light) in lights.iter().enumerate() {
            if let Some(emitter) = light_emitter(light, index, LightTreeSettings::default()) {
                emitters.push(emitter);
                indices.push(index);
            }
        }
        assert_eq!(emitters.len(), lights.len());
        let local = build_node(&emitters, indices, false);
        let distant = build_node(&emitters, Vec::new(), true);
        let root = Node {
            measure: combine_measure(local.measure, distant.measure),
            kind: NodeKind::Inner(Box::new(local), Box::new(distant)),
        };
        let mut paths = vec![(0_u64, 0_u32); emitters.len()];
        collect_paths(&root, 0, 0, &mut paths);
        LightTree {
            lamp_emitters: (0..lights.len()).map(Some).collect(),
            world_emitter: None,
            mesh_subtrees: Vec::new(),
            lights,
            root: Some(root),
            paths,
            emitters,
        }
    }
    #[test]
    fn unnormalized_light_tree_energy_includes_cycles_area() {
        let settings = LightTreeSettings {
            normalize: false,
            strength_scale: DVec3::splat(2.0),
        };
        let mut point = point_light(DVec3::new(0.0, 0.0, 2.0), 1.0);
        point.radius = 0.25;
        let point_energy = light_emitter(&point, 0, settings).map(|emitter| emitter.measure.energy);
        assert!(point_energy.is_some_and(|energy| {
            (energy - 2.0 * point.radius * point.radius).abs() < 1.0e-12
        }));

        let mut area = point_light(DVec3::ZERO, 1.0);
        area.kind = LightType::Area;
        let area_shape =
            crate::render::light_model::AreaShape::new(DVec3::X, DVec3::Y, 0.5, 0.25, false);
        area.area = Some(area_shape);
        let area_energy = light_emitter(&area, 0, settings).map(|emitter| emitter.measure.energy);
        assert!(
            area_energy
                .is_some_and(|energy| { (energy - area_shape.area * 2.0 / PI).abs() < 1.0e-12 })
        );

        let mut sun = point_light(DVec3::ZERO, 1.0);
        sun.kind = LightType::Sun;
        sun.spot_size = 0.6;
        let sun_energy = light_emitter(&sun, 0, settings).map(|emitter| emitter.measure.energy);
        let expected_sun = 2.0 * PI * (sun.spot_size * 0.5).sin().powi(2);
        assert!(sun_energy.is_some_and(|energy| (energy - expected_sun).abs() < 1.0e-12));
    }

    #[test]
    fn world_energy_uses_shaded_average_radiance_and_emission_estimate() {
        let doc = SceneDoc::new("00000000-0000-4000-8000-000000000001".to_owned());
        let world = World {
            color: [0.05; 3],
            strength: 2.0,
            ..World::default()
        };
        let emitter = world_emitter(&doc, &world, 0.1);
        assert!((emitter.measure.energy - 0.01 * PI).abs() < 1.0e-12);
    }

    fn query(volume: bool) -> Query {
        Query {
            point: DVec3::ZERO,
            normal_or_direction: if volume { DVec3::X } else { DVec3::Z },
            distance: 10.0,
            volume,
            has_transmission: volume,
        }
    }

    #[test]
    fn equal_emitters_have_equal_selection_pmf_and_sum_to_one() {
        let tree = point_tree(vec![
            point_light(DVec3::new(0.0, 0.0, 2.0), 1.0),
            point_light(DVec3::new(0.0, 0.0, 2.0), 1.0),
        ]);
        for query in [query(false), query(true)] {
            let pdfs: Vec<f64> = (0..2).map(|index| tree.pdf(index, query)).collect();
            assert!((pdfs.iter().sum::<f64>() - 1.0).abs() < 1.0e-12);
            assert!((pdfs[0] - 0.5).abs() < 1.0e-12);
            assert!((pdfs[1] - 0.5).abs() < 1.0e-12);
        }
    }
    #[test]
    fn volume_area_importance_includes_emitter_behind_segment() {
        let mut light = point_light(DVec3::new(0.0, 0.0, -2.0), 1.0);
        light.kind = LightType::Area;
        light.direction = DVec3::Z;
        light.area = Some(crate::render::light_model::AreaShape::new(
            DVec3::X,
            DVec3::Y,
            0.5,
            0.5,
            false,
        ));
        let mut query = query(true);
        query.normal_or_direction = DVec3::Z;
        let importance = light_emitter(&light, 0, LightTreeSettings::default())
            .map(|emitter| emitter_importance(emitter, &[light], &[], query));

        assert!(importance.is_some_and(|importance| importance.max > 0.0));
    }

    #[test]
    fn auto_emissive_triangles_use_cycles_double_sided_cones() {
        let positions = [DVec3::ZERO, DVec3::X, DVec3::Y];
        let normal = DVec3::Z;
        let auto = emissive_triangle_cone(EmissionSampling::Auto, positions, normal);
        let front_back = emissive_triangle_cone(EmissionSampling::FrontBack, positions, normal);
        assert_eq!(auto.axis, front_back.axis);
        assert_eq!(auto.outer, front_back.outer);
        assert_eq!(auto.emission, front_back.emission);
    }

    #[test]
    fn leaf_pmf_matches_cycles_when_minimum_importance_is_positive() {
        let tree = point_tree(vec![
            point_light(DVec3::new(0.0, 0.0, 2.0), 4.0),
            point_light(DVec3::new(0.0, 0.0, 2.0), 1.0),
        ]);
        // With identical point-light geometry, Cycles' minimum and maximum importance are equal.
        let first_importance = 4.0;
        let second_importance = 1.0;
        let expected_first = first_importance / (first_importance + second_importance);
        let pdfs: Vec<f64> = (0..2).map(|index| tree.pdf(index, query(false))).collect();
        assert!((pdfs[0] - expected_first).abs() < 1.0e-12);
        assert!((pdfs.iter().sum::<f64>() - 1.0).abs() < 1.0e-12);
    }

    #[test]
    fn emissive_mesh_triangles_form_a_subtree_and_change_other_emitter_pmfs() {
        let lights = vec![
            point_light(DVec3::new(0.0, 0.0, 2.0), 1.0),
            point_light(DVec3::new(0.0, 0.0, 2.0), 1.0),
        ];
        let baseline = point_tree(lights.clone());
        let baseline_pdf = baseline.pdf(0, query(false));
        let mut tree = point_tree(lights);
        let triangle_measure = |x: f64, energy: f64| {
            let positions = [
                DVec3::new(x, 0.0, 2.0),
                DVec3::new(x + 1.0, 0.0, 2.0),
                DVec3::new(x, 1.0, 2.0),
            ];
            let mut bounds = Bounds::EMPTY;
            for point in positions {
                bounds.grow(point);
            }
            TriangleMeasure {
                centroid: (positions[0] + positions[1] + positions[2]) / 3.0,
                measure: Measure {
                    bounds,
                    cone: Cone {
                        axis: DVec3::Z,
                        outer: 0.0,
                        emission: PI * 0.5,
                    },
                    energy,
                },
            }
        };
        let subtree = build_mesh_subtree(vec![
            triangle_measure(3.0, 2.0),
            triangle_measure(5.0, 3.0),
            triangle_measure(7.0, 4.0),
        ]);
        assert_eq!(subtree.emitters.len(), 3);
        assert!(
            subtree
                .emitters
                .iter()
                .all(|emitter| matches!(emitter.kind, EmitterKind::MeshTriangle))
        );
        let measure = subtree.root.measure;
        let centroid = measure.bounds.center();
        let subtree_index = tree.mesh_subtrees.len();
        tree.mesh_subtrees.push(subtree);
        tree.emitters.push(Emitter {
            kind: EmitterKind::Mesh(subtree_index),
            centroid,
            importance_centroid: centroid,
            measure,
        });
        let local = build_node(&tree.emitters, vec![0, 1, 2], false);
        let distant = build_node(&tree.emitters, Vec::new(), true);
        let root = Node {
            measure: combine_measure(local.measure, distant.measure),
            kind: NodeKind::Inner(Box::new(local), Box::new(distant)),
        };
        tree.paths = vec![(0_u64, 0_u32); tree.emitters.len()];
        collect_paths(&root, 0, 0, &mut tree.paths);
        tree.root = Some(root);
        let pmfs: Vec<f64> = (0..tree.emitters.len())
            .map(|index| tree.pdf(index, query(false)))
            .collect();
        assert!((pmfs.iter().sum::<f64>() - 1.0).abs() < 1.0e-12);
        assert!((pmfs[0] - baseline_pdf).abs() > 1.0e-3);
    }

    #[test]
    fn cone_measure_matches_a_point_emitter_hemisphere() {
        let cone = Cone {
            axis: DVec3::Z,
            outer: PI,
            emission: PI * 0.5,
        };
        assert!((cone.measure() - 4.0 * PI).abs() < 1.0e-12);
    }

    #[test]
    fn disabled_tree_uses_the_flat_selection_pmf() {
        let selection = super::super::LightSelectionPdfs {
            lamps: 0.25,
            world: 0.125,
            tree: None,
        };
        let query = super::super::LightTreeQuery {
            point: DVec3::ZERO,
            normal_or_direction: DVec3::Z,
            distance: 0.0,
            volume: false,
            has_transmission: false,
            clip_to_light: false,
        };
        assert_eq!(selection.lamp_pdf(7, query), 0.25);
        assert_eq!(selection.world_pdf(query), 0.125);
        let mut probabilities = vec![1.0];
        selection.lamp_pdfs(query, &mut probabilities);
        assert_eq!(probabilities, Vec::<f64>::new());
    }
}
