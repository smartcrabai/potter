//! Blender 5.2.2 bevel parity over the checked-in corpus (`tests/fixtures/bevel_parity_oracle.*`).
//!
//! Every case is evaluated and compared; nothing stops at the first failing case. Each case
//! yields one row of the printed table.
//!
//! Expected rows are canonicalized so storage order does not matter: vertices sort numerically by
//! (position, group weights), edges sort by (canonical ids, seam, sharp, weight), each face ring
//! starts at its lexicographic minimum rotation over (position, canonical id), and faces sort by
//! (ring ids, material, smooth, strength). Coincident vertices stay separate rows; evaluated
//! vertices are associated with them by topology in `match_vertices`, never merged.
//!
//! Tolerances:
//! - positions, group weights and UVs: 1e-6 absolute. The oracle stores binary32 values and Potter
//!   computes in binary64, so the comparison is against the binary32 rounding of Blender's output.
//! - corner normals: [`NORMAL_TOLERANCE`], the bound on Blender's 16-bit custom-normal codes.
//! - topology, edge and face flags, material indices and face strength: exact.
//!
//! The UV-layer evaluation tests at the end of this file check that bevel on `uv_layers` evaluates
//! and keeps a UV row per face corner; the oracle compares the values.

use std::{
    cmp::Ordering,
    collections::{BTreeMap, BTreeSet, HashMap, HashSet},
    error::Error,
    panic::{AssertUnwindSafe, catch_unwind},
};

use glam::DVec3;
use potter_core::{
    error::PotError,
    geom::{Mesh, corner_normal_vectors, modifiers::evaluate_modifiers},
    model::{Id, Modifier, ModifierRuntime},
};
use serde_json::{Map, Value, json};

const ORACLE: &str = include_str!("fixtures/bevel_parity_oracle.json");
const POSITION_TOLERANCE: f64 = 1.0e-6;
const WEIGHT_TOLERANCE: f64 = 1.0e-6;
const UV_TOLERANCE: f64 = 1.0e-6;
/// Blender's setter can snap restored normals to a different reference direction; this fixture
/// bound validates that probe only and is not the Potter-versus-Blender comparison tolerance.
const BLENDER_RESTORE_ERROR_LIMIT: f64 = 0.014_238_2;
/// Corner normals must agree within the measured codec-level residual, not the setter's snap limit.
const NORMAL_TOLERANCE: f64 = 2.0e-4;

type TestResult<T = ()> = Result<T, Box<dyn Error>>;

#[test]
fn every_bevel_oracle_case_matches_blender_geometry_and_attributes() -> TestResult {
    let oracle: Value = serde_json::from_str(ORACLE)?;
    assert_eq!(oracle["schema_version"], json!(2));
    assert_eq!(oracle["blender_version"], json!("5.2.2 LTS"));
    assert_eq!(
        oracle["float_encoding"],
        json!("little-endian IEEE-754 binary32 hex")
    );
    let measured = oracle["normal_quantization"]["stored_restore_max_angle_error_radians"]
        .as_f64()
        .ok_or("normal_quantization.stored_restore_max_angle_error_radians must be a number")?;
    assert!(
        measured <= BLENDER_RESTORE_ERROR_LIMIT,
        "measured custom-normal restore error {measured} exceeds its probe bound {BLENDER_RESTORE_ERROR_LIMIT}"
    );
    let cases = oracle["cases"]
        .as_array()
        .ok_or("oracle cases must be an array")?;
    assert!(
        !cases.is_empty(),
        "the checked-in bevel oracle must contain cases"
    );

    let reports: Vec<CaseReport> = cases.iter().map(|case| run_case(&oracle, case)).collect();
    print_table(&reports);

    let mut unexplained = Vec::new();
    for report in &reports {
        let columns = report.differing_columns();
        match KNOWN_GAPS.iter().find(|gap| gap.case == report.name) {
            Some(gap) if columns != gap.columns => unexplained.push(format!(
                "{}: differs in {columns:?}, recorded gap is {:?} ({}); fixed or changed, so update KNOWN_GAPS",
                report.name, gap.columns, gap.cause
            )),
            None if !columns.is_empty() => unexplained.push(format!(
                "{}: differs in {columns:?} and is not a recorded gap",
                report.name
            )),
            _ => {}
        }
    }
    for gap in KNOWN_GAPS {
        if !reports.iter().any(|report| report.name == gap.case) {
            unexplained.push(format!("{}: recorded gap names no corpus case", gap.case));
        }
    }
    if unexplained.is_empty() {
        return Ok(());
    }
    Err(format!(
        "{} bevel oracle issues: {}",
        unexplained.len(),
        unexplained.join("; ")
    )
    .into())
}

/// A measured divergence from Blender 5.2.2 that is not yet fixed. `columns` are the comparison columns
/// that still differ, and the test asserts they stay exactly that: a fix or a regression both fail until
/// the entry is updated or removed.
struct KnownGap {
    case: &'static str,
    columns: &'static [&'static str],
    cause: &'static str,
}

/// Every corpus case that still differs from Blender, with the columns and a one-line cause.
const KNOWN_GAPS: &[KnownGap] = &[
    KnownGap {
        case: "cube_s2_p100_vertices_absolute",
        columns: &["nrm"],
        cause: "flat faces of non-planar clamped quads: Blender's corner normal is neither the Newell nor the diagonal normal of the output positions",
    },
    KnownGap {
        case: "clamp_large_offset_on",
        columns: &["nrm"],
        cause: "same non-planar flat-face normal source as cube_s2_p100_vertices_absolute",
    },
    KnownGap {
        case: "clamp_small_width_on",
        columns: &["nrm"],
        cause: "normal residual 2.7e-4 above tolerance 2e-4 on clamped width corners; not analysed",
    },
    KnownGap {
        case: "clamp_small_width_off",
        columns: &["nrm"],
        cause: "normal residual 2.7e-4 above tolerance 2e-4 on clamped width corners; not analysed",
    },
    KnownGap {
        case: "cube_s5_p80_vertices_offset",
        columns: &["nrm"],
        cause: "normal residual 4.2e-4 above tolerance 2e-4 on vertex-mode 80% offset corners; not analysed",
    },
    KnownGap {
        case: "oblique_prism_s4_p80",
        columns: &["nrm"],
        cause: "normal residual 2.9e-4 above tolerance 2e-4; not analysed",
    },
    KnownGap {
        case: "clamp_large_offset_off",
        columns: &["wgt"],
        cause: "vertex weight 0.51 at clamped corners; not analysed",
    },
    KnownGap {
        case: "clamp_large_width_off",
        columns: &["wgt"],
        cause: "vertex weight 0.16 at clamped corners; not analysed",
    },
    KnownGap {
        case: "cube_s3_p25_vertices_absolute",
        columns: &["wgt", "uv"],
        cause: "vertex-mode odd-segment corner snaps onto a different input edge at the same vertex than Blender (localized by per-writer dump; not fixed)",
    },
    KnownGap {
        case: "cube_s5_p25_vertices_width",
        columns: &["wgt", "uv"],
        cause: "same vertex-mode corner snap as cube_s3_p25_vertices_absolute",
    },
    KnownGap {
        case: "nonmanifold_three_faces_s5_p0",
        columns: &["wgt", "uv"],
        cause: "vertex-mode corner snap, four points no corner writes, and a vertex UV blend no single face reproduces",
    },
    KnownGap {
        case: "pentagonal_pyramid_s3_adj_vertices",
        columns: &["wgt", "uv"],
        cause: "vertex-mode odd-segment corners, the cube_s3_p25 family; not localized per case",
    },
    KnownGap {
        case: "pentagonal_pyramid_s3_cutoff_vertices",
        columns: &["wgt", "uv"],
        cause: "vertex-mode odd-segment corners, the cube_s3_p25 family; not localized per case",
    },
    KnownGap {
        case: "hexagonal_pyramid_s3_adj_vertices",
        columns: &["wgt", "uv"],
        cause: "vertex-mode odd-segment corners, the cube_s3_p25 family; not localized per case",
    },
    KnownGap {
        case: "hexagonal_pyramid_s3_cutoff_vertices",
        columns: &["wgt", "uv"],
        cause: "vertex-mode odd-segment corners, the cube_s3_p25 family; not localized per case",
    },
    KnownGap {
        case: "irregular_octahedron_s3_adj_vertices",
        columns: &["wgt", "uv"],
        cause: "vertex-mode odd-segment corners, the cube_s3_p25 family; not localized per case",
    },
    KnownGap {
        case: "irregular_octahedron_s3_cutoff_vertices",
        columns: &["wgt", "uv"],
        cause: "vertex-mode odd-segment corners, the cube_s3_p25 family; not localized per case",
    },
    KnownGap {
        case: "hexagonal_pyramid_mark_sharp_vertices_s3",
        columns: &["wgt", "uv"],
        cause: "vertex-mode odd-segment corners, the cube_s3_p25 family; not localized per case",
    },
    KnownGap {
        case: "open_fan_vertices_cutoff_s3",
        columns: &["wgt", "uv"],
        cause: "vertex-mode odd-segment corners, the cube_s3_p25 family; not localized per case",
    },
];

/// Runs one case. Errors and panics become a failing report rather than aborting the run.
fn run_case(oracle: &Value, case: &Value) -> CaseReport {
    let name = case["name"].as_str().unwrap_or("<unnamed>").to_owned();
    match catch_unwind(AssertUnwindSafe(|| compare_case(oracle, case, &name))) {
        Ok(Ok(report)) => report,
        Ok(Err(error)) => CaseReport::errored(&name, error.to_string()),
        Err(payload) => CaseReport::errored(&name, panic_message(payload.as_ref())),
    }
}

fn panic_message(payload: &(dyn std::any::Any + Send)) -> String {
    if let Some(text) = payload.downcast_ref::<&str>() {
        return format!("panic: {text}");
    }
    if let Some(text) = payload.downcast_ref::<String>() {
        return format!("panic: {text}");
    }
    "panic with a non-string payload".to_owned()
}

fn compare_case(oracle: &Value, case: &Value, name: &str) -> TestResult<CaseReport> {
    let shape = case["shape"]
        .as_str()
        .ok_or("oracle case shape must be a string")?;
    let input = oracle["inputs"]
        .get(shape)
        .ok_or_else(|| format!("oracle input {shape} is missing"))?;
    let mesh = input_mesh(input)?;
    let evaluated = evaluate_bevel(&mesh, &case["parameters"])??;
    let expected = expected_mesh(&case["output"])?;
    let actual = actual_mesh(&evaluated, &expected.group_names)?;

    compare(name, &canonicalize(&expected), &actual)
}

// ---------------------------------------------------------------------------------------------
// Raw meshes: the oracle output and Potter's evaluated mesh, before canonicalization.
// ---------------------------------------------------------------------------------------------

struct RawMesh {
    group_names: Vec<String>,
    vertices: Vec<RawVertex>,
    edges: Vec<RawEdge>,
    faces: Vec<RawFace>,
}

struct RawVertex {
    position: DVec3,
    weights: Vec<f64>,
}

struct RawEdge {
    vertices: [usize; 2],
    seam: bool,
    sharp: bool,
    weight: Option<f64>,
}

struct RawFace {
    material: i64,
    smooth: bool,
    strength: Option<i64>,
    ring: Vec<RawCorner>,
}

#[derive(Clone, Copy)]
struct RawCorner {
    vertex: usize,
    uv: [f64; 2],
    normal: DVec3,
}

fn array<'a>(value: &'a Value, context: &'static str) -> TestResult<&'a Vec<Value>> {
    value.as_array().ok_or_else(|| context.into())
}

fn index_value(value: &Value, context: &'static str) -> TestResult<usize> {
    Ok(usize::try_from(value.as_u64().ok_or(context)?)?)
}

/// One binary32 value from its little-endian hex encoding, as an exact binary64 value.
fn float_bits(value: &Value) -> TestResult<f64> {
    let text = value.as_str().ok_or("float bits must be a hex string")?;
    let bytes: [u8; 4] = hex::decode(text)?
        .try_into()
        .map_err(|_| "float bits must encode four bytes")?;
    Ok(f64::from(f32::from_le_bytes(bytes)))
}

fn bit_array(value: &Value, context: &'static str) -> TestResult<Vec<f64>> {
    array(value, context)?.iter().map(float_bits).collect()
}

fn bit_triple(value: &Value, context: &'static str) -> TestResult<[f64; 3]> {
    bit_array(value, context)?
        .try_into()
        .map_err(|_| "a position or normal must have three components".into())
}

fn bit_pair(value: &Value, context: &'static str) -> TestResult<[f64; 2]> {
    bit_array(value, context)?
        .try_into()
        .map_err(|_| "a UV must have two components".into())
}

fn expected_mesh(output: &Value) -> TestResult<RawMesh> {
    let group_names = array(
        &output["group_names"],
        "oracle group names must be an array",
    )?
    .iter()
    .map(|name| {
        name.as_str()
            .map(str::to_owned)
            .ok_or_else(|| "oracle group name must be a string".into())
    })
    .collect::<TestResult<Vec<String>>>()?;
    let vertices = array(&output["vertices"], "oracle vertices must be an array")?
        .iter()
        .map(|row| {
            let row = array(row, "oracle vertex row must be an array")?;
            let [position, weights] = row.as_slice() else {
                return Err("oracle vertex row must have position and weights".into());
            };
            Ok(RawVertex {
                position: DVec3::from_array(bit_triple(position, "oracle position")?),
                weights: bit_array(weights, "oracle group weights")?,
            })
        })
        .collect::<TestResult<Vec<_>>>()?;
    let edges = array(&output["edges"], "oracle edges must be an array")?
        .iter()
        .map(|row| {
            let row = array(row, "oracle edge row must be an array")?;
            let [first, second, seam, sharp, weight] = row.as_slice() else {
                return Err("oracle edge row must have five fields".into());
            };
            Ok(RawEdge {
                vertices: [
                    index_value(first, "oracle edge vertex")?,
                    index_value(second, "oracle edge vertex")?,
                ],
                seam: seam.as_bool().ok_or("oracle edge seam must be boolean")?,
                sharp: sharp
                    .as_bool()
                    .ok_or("oracle edge sharp flag must be boolean")?,
                weight: if weight.is_null() {
                    None
                } else {
                    Some(float_bits(weight)?)
                },
            })
        })
        .collect::<TestResult<Vec<_>>>()?;
    let faces = array(&output["faces"], "oracle faces must be an array")?
        .iter()
        .map(|row| {
            let row = array(row, "oracle face row must be an array")?;
            let [material, smooth, strength, ring] = row.as_slice() else {
                return Err("oracle face row must have four fields".into());
            };
            let ring = array(ring, "oracle face ring must be an array")?
                .iter()
                .map(|corner| {
                    let corner = array(corner, "oracle corner must be an array")?;
                    let [vertex, u, v, nx, ny, nz] = corner.as_slice() else {
                        return Err("oracle corner must have six fields".into());
                    };
                    Ok(RawCorner {
                        vertex: index_value(vertex, "oracle corner vertex")?,
                        uv: bit_pair(&json!([u, v]), "oracle UV")?,
                        normal: DVec3::from_array(bit_triple(
                            &json!([nx, ny, nz]),
                            "oracle normal",
                        )?),
                    })
                })
                .collect::<TestResult<Vec<_>>>()?;
            Ok(RawFace {
                material: material
                    .as_i64()
                    .ok_or("oracle face material must be an integer")?,
                smooth: smooth
                    .as_bool()
                    .ok_or("oracle face smooth flag must be boolean")?,
                strength: if strength.is_null() {
                    None
                } else {
                    Some(
                        strength
                            .as_i64()
                            .ok_or("oracle face strength must be an integer")?,
                    )
                },
                ring,
            })
        })
        .collect::<TestResult<Vec<_>>>()?;
    Ok(RawMesh {
        group_names,
        vertices,
        edges,
        faces,
    })
}

/// Potter's evaluated mesh in the same raw form. Attribute readers mirror native storage: face
/// smoothness and edge seam/sharp flags are keyed by persistent IDs, UVs use the `uv_layers` store,
/// and bevel weights are keyed `float` attributes.
fn actual_mesh(mesh: &Mesh, expected_groups: &[String]) -> TestResult<RawMesh> {
    let group_names = vertex_group_names(mesh);
    let expected_set: HashSet<&str> = expected_groups.iter().map(String::as_str).collect();
    let actual_set: HashSet<&str> = group_names.iter().map(String::as_str).collect();
    if expected_set != actual_set {
        return Err(
            format!("evaluated vertex groups {actual_set:?} differ from {expected_set:?}").into(),
        );
    }
    let index_by_id: HashMap<u32, usize> = mesh
        .vertices
        .iter()
        .enumerate()
        .map(|(index, vertex)| (vertex.id, index))
        .collect();
    let normals = corner_normal_vectors(mesh)?;
    let uvs = uv_layer_values(mesh)?;
    let corner_count: usize = mesh.faces.iter().map(|face| face.vertices.len()).sum();
    if normals.len() != corner_count || uvs.len() != corner_count {
        return Err("evaluated corner normal or UV count differs from the corner count".into());
    }

    let vertices = mesh
        .vertices
        .iter()
        .map(|vertex| {
            Ok(RawVertex {
                position: vertex.co,
                weights: expected_groups
                    .iter()
                    .map(|group| group_weight(mesh, group, vertex.id))
                    .collect::<TestResult<Vec<_>>>()?,
            })
        })
        .collect::<TestResult<Vec<_>>>()?;

    let mut edges = Vec::with_capacity(mesh.edges.len());
    for edge in &mesh.edges {
        let key = format!("e{}", edge.id);
        let seam = keyed_flag(mesh, "seam", &key).unwrap_or(false);
        let sharp = keyed_flag(mesh, "sharp", &key).unwrap_or(false);
        let weight = keyed_float(mesh, "bevel_weight_edge", &key, "edge")?;
        edges.push(RawEdge {
            vertices: [
                *index_by_id
                    .get(&edge.vertices[0])
                    .ok_or("edge endpoint is missing")?,
                *index_by_id
                    .get(&edge.vertices[1])
                    .ok_or("edge endpoint is missing")?,
            ],
            seam,
            sharp,
            weight,
        });
    }

    let strengths = mesh.attributes.get("__mod_weightednormals_faceweight");
    let mut faces = Vec::with_capacity(mesh.faces.len());
    let mut corner = 0;
    for face in &mesh.faces {
        let smooth = keyed_flag(mesh, "shade_smooth", &format!("f{}", face.id)).unwrap_or(false);
        let strength = match strengths {
            None => None,
            // Blender's strength constants (16384, 0, -16384) are stored as written, one per face.
            Some(attribute) => Some(
                attribute
                    .get("values")
                    .and_then(Value::as_object)
                    .and_then(|values| values.get(&format!("f{}", face.id)))
                    .and_then(Value::as_i64)
                    .ok_or("face strength must be an integer for every face")?,
            ),
        };
        let mut ring = Vec::with_capacity(face.vertices.len());
        for vertex_id in &face.vertices {
            ring.push(RawCorner {
                vertex: *index_by_id.get(vertex_id).ok_or("face vertex is missing")?,
                uv: uvs[corner],
                normal: normals[corner],
            });
            corner += 1;
        }
        faces.push(RawFace {
            material: i64::from(face.material_index),
            smooth,
            strength,
            ring,
        });
    }
    Ok(RawMesh {
        group_names: expected_groups.to_vec(),
        vertices,
        edges,
        faces,
    })
}

fn vertex_group_names(mesh: &Mesh) -> Vec<String> {
    let groups = mesh
        .attributes
        .get("vertex_groups")
        .and_then(Value::as_object);
    groups
        .map(|groups| groups.keys().cloned().collect())
        .unwrap_or_default()
}

fn group_weight(mesh: &Mesh, group: &str, vertex_id: u32) -> TestResult<f64> {
    let weights = mesh
        .attributes
        .get("vertex_groups")
        .and_then(Value::as_object)
        .and_then(|groups| groups.get(group))
        .and_then(Value::as_object);
    match weights.and_then(|weights| weights.get(&format!("v{vertex_id}"))) {
        None => Ok(0.0),
        Some(value) => Ok(value
            .as_f64()
            .ok_or("vertex group weight must be numeric")?),
    }
}

fn keyed_flag(mesh: &Mesh, name: &str, key: &str) -> Option<bool> {
    mesh.attributes
        .get(name)?
        .get("values")?
        .get(key)?
        .as_bool()
}
fn keyed_float(mesh: &Mesh, name: &str, key: &str, domain: &str) -> TestResult<Option<f64>> {
    let Some(attribute) = mesh.attributes.get(name) else {
        return Ok(None);
    };
    if attribute.get("domain").and_then(Value::as_str) != Some(domain)
        || attribute.get("type").and_then(Value::as_str) != Some("float")
    {
        return Err(format!("evaluated {name} must be a {domain} float attribute").into());
    }
    // Potter writes no entry for an element without a source; a reader takes the FLOAT default, 0.0
    // (`provenance::default_value`), which Blender also holds for the new edges.
    let Some(value) = attribute
        .get("values")
        .and_then(Value::as_object)
        .and_then(|values| values.get(key))
    else {
        return Ok(Some(0.0));
    };
    Ok(Some(value.as_f64().ok_or_else(|| {
        format!("evaluated {name} value {key} is not numeric")
    })?))
}

fn uv_layer_values(mesh: &Mesh) -> TestResult<Vec<[f64; 2]>> {
    let layer = mesh
        .attributes
        .get("uv_layers")
        .and_then(|store| store.get("layers"))
        .and_then(Value::as_array)
        .and_then(|layers| {
            layers
                .iter()
                .find(|layer| layer.get("name").and_then(Value::as_str) == Some("UVMap"))
        })
        .ok_or("evaluated mesh lost UVMap")?;
    let faces = array(&layer["faces"], "evaluated UVMap faces must be an array")?;
    let mut rows = HashMap::with_capacity(faces.len());
    for face in faces {
        let face_id = face["face_id"]
            .as_u64()
            .ok_or("evaluated UV face ID must be unsigned")?;
        if rows.insert(face_id, &face["uv"]).is_some() {
            return Err("evaluated UVMap contains a duplicate face ID".into());
        }
    }

    let corner_count: usize = mesh.faces.iter().map(|face| face.vertices.len()).sum();
    let mut values = Vec::with_capacity(corner_count);
    for face in &mesh.faces {
        let row = rows
            .get(&u64::from(face.id))
            .ok_or("evaluated UVMap is missing a face")?;
        let corners = array(row, "evaluated UV face values must be an array")?;
        if corners.len() != face.vertices.len() {
            return Err("evaluated UV face value count differs from its corner count".into());
        }
        for value in corners {
            let pair = array(value, "evaluated UV corner must be an array")?;
            let [u, v] = pair.as_slice() else {
                return Err("evaluated UV corner must have two values".into());
            };
            // The oracle stores Blender's V-up values; compare in that convention after reading
            // the canonical store, whose values are V-down.
            values.push([
                u.as_f64().ok_or("evaluated UV must be numeric")?,
                1.0 - v.as_f64().ok_or("evaluated UV must be numeric")?,
            ]);
        }
    }
    Ok(values)
}

// ---------------------------------------------------------------------------------------------
// Canonical form, shared by both sides.
// ---------------------------------------------------------------------------------------------

/// A binary64 value ordered by `total_cmp`. Canonical values are normalised by [`quantize`], so
/// equal values also compare equal.
#[derive(Clone, Copy, Debug)]
struct Ordered(f64);

impl PartialEq for Ordered {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other).is_eq()
    }
}

impl Eq for Ordered {}

impl PartialOrd for Ordered {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Ordered {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.total_cmp(&other.0)
    }
}

/// The binary32 rounding of `value`. Adding `0.0` turns `-0.0` into `0.0`, so that equal
/// positions and weights compare equal, as Blender's float32 bit patterns do after normalisation.
fn quantize(value: f64) -> f64 {
    f64::from(value as f32) + 0.0
}

type VertexKey = Vec<Ordered>;

struct CanonVertex {
    key: VertexKey,
    position: DVec3,
    weights: Vec<f64>,
}

struct CanonEdge {
    vertices: [usize; 2],
    seam: bool,
    sharp: bool,
    weight: Option<f64>,
}

struct CanonCorner {
    vertex: usize,
    uv: [f64; 2],
    normal: DVec3,
}

struct CanonFace {
    material: i64,
    smooth: bool,
    strength: Option<i64>,
    ring: Vec<CanonCorner>,
}

struct CanonMesh {
    vertices: Vec<CanonVertex>,
    edges: Vec<CanonEdge>,
    faces: Vec<CanonFace>,
}

fn canonicalize(raw: &RawMesh) -> CanonMesh {
    let keys: Vec<VertexKey> = raw
        .vertices
        .iter()
        .map(|vertex| {
            vertex
                .position
                .to_array()
                .into_iter()
                .chain(vertex.weights.iter().copied())
                .map(|value| Ordered(quantize(value)))
                .collect()
        })
        .collect();
    let mut order: Vec<usize> = (0..raw.vertices.len()).collect();
    order.sort_by(|&left, &right| keys[left].cmp(&keys[right]).then(left.cmp(&right)));
    // Coincident rows stay separate vertices: merging them would erase output distinctions. Their
    // labels follow the stored order and only name them; `match_vertices` assigns them by topology.
    let mut vertices: Vec<CanonVertex> = Vec::with_capacity(raw.vertices.len());
    let mut canonical_of = vec![0_usize; raw.vertices.len()];
    for (label, &index) in order.iter().enumerate() {
        vertices.push(CanonVertex {
            key: keys[index].clone(),
            position: raw.vertices[index].position,
            weights: raw.vertices[index].weights.clone(),
        });
        canonical_of[index] = label;
    }

    let mut edges: Vec<CanonEdge> = raw
        .edges
        .iter()
        .map(|edge| {
            let first = canonical_of[edge.vertices[0]];
            let second = canonical_of[edge.vertices[1]];
            CanonEdge {
                vertices: [first.min(second), first.max(second)],
                seam: edge.seam,
                sharp: edge.sharp,
                weight: edge.weight,
            }
        })
        .collect();
    edges.sort_by_key(edge_order);

    let mut faces: Vec<CanonFace> = raw
        .faces
        .iter()
        .map(|face| {
            let mut ring: Vec<CanonCorner> = face
                .ring
                .iter()
                .map(|corner| CanonCorner {
                    vertex: canonical_of[corner.vertex],
                    uv: corner.uv,
                    normal: corner.normal,
                })
                .collect();
            rotate_to_minimum(&mut ring, &vertices);
            CanonFace {
                material: face.material,
                smooth: face.smooth,
                strength: face.strength,
                ring,
            }
        })
        .collect();
    faces.sort_by_key(face_order);
    CanonMesh {
        vertices,
        edges,
        faces,
    }
}

type EdgeOrder = ([usize; 2], bool, bool, Option<Ordered>);

fn edge_order(edge: &CanonEdge) -> EdgeOrder {
    (
        edge.vertices,
        edge.seam,
        edge.sharp,
        edge.weight.map(|weight| Ordered(quantize(weight))),
    )
}

type FaceOrder = (Vec<usize>, i64, bool, Option<i64>);

fn face_order(face: &CanonFace) -> FaceOrder {
    (
        face.ring.iter().map(|corner| corner.vertex).collect(),
        face.material,
        face.smooth,
        face.strength,
    )
}

/// Rotates the ring to the lexicographic minimum over (position, canonical id) corner keys.
fn rotate_to_minimum(ring: &mut [CanonCorner], vertices: &[CanonVertex]) {
    let count = ring.len();
    let keys: Vec<([Ordered; 3], usize)> = ring
        .iter()
        .map(|corner| {
            let key = &vertices[corner.vertex].key;
            ([key[0], key[1], key[2]], corner.vertex)
        })
        .collect();
    let start = (0..count).min_by(|&left, &right| {
        (0..count)
            .map(|step| keys[(left + step) % count])
            .cmp((0..count).map(|step| keys[(right + step) % count]))
    });
    if let Some(start) = start {
        ring.rotate_left(start);
    }
}

// ---------------------------------------------------------------------------------------------
// Comparison and reporting.
// ---------------------------------------------------------------------------------------------

struct CaseReport {
    name: String,
    error: Option<String>,
    expected_vertices: usize,
    actual_vertices: usize,
    position_bad: usize,
    position_max: f64,
    weight_bad: usize,
    weight_max: f64,
    uv_bad: usize,
    uv_max: f64,
    normal_bad: usize,
    normal_max: f64,
    ring_bad: usize,
    edge_bad: usize,
    flag_bad: usize,
    /// Evaluated vertices left tied between non-interchangeable expected vertices. Each is also counted in `position_bad`.
    ambiguous: usize,
}

impl CaseReport {
    fn new(name: &str, expected_vertices: usize, actual_vertices: usize) -> Self {
        Self {
            name: name.to_owned(),
            error: None,
            expected_vertices,
            actual_vertices,
            position_bad: 0,
            position_max: 0.0,
            weight_bad: 0,
            weight_max: 0.0,
            uv_bad: 0,
            uv_max: 0.0,
            normal_bad: 0,
            normal_max: 0.0,
            ring_bad: 0,
            edge_bad: 0,
            flag_bad: 0,
            ambiguous: 0,
        }
    }

    fn errored(name: &str, error: String) -> Self {
        Self {
            error: Some(error),
            ..Self::new(name, 0, 0)
        }
    }

    /// The comparison columns that differ from Blender, in table order. A case that errors or has the wrong
    /// vertex count reports `error` or `count`.
    fn differing_columns(&self) -> Vec<&'static str> {
        if self.error.is_some() {
            return vec!["error"];
        }
        let mut columns = Vec::new();
        if self.expected_vertices != self.actual_vertices {
            columns.push("count");
        }
        for (name, bad) in [
            ("pos", self.position_bad),
            ("wgt", self.weight_bad),
            ("uv", self.uv_bad),
            ("nrm", self.normal_bad),
            ("ring", self.ring_bad),
            ("edge", self.edge_bad),
            ("flag", self.flag_bad),
        ] {
            if bad > 0 {
                columns.push(name);
            }
        }
        columns
    }

    fn failed(&self) -> bool {
        !self.differing_columns().is_empty()
    }
}

fn max_abs_difference(left: DVec3, right: DVec3) -> f64 {
    (left - right).abs().max_element()
}

/// Groups rows by key, keeping each key's rows in their (already canonical) order.
fn group_by<'a, T, K: Ord + Clone>(
    rows: &'a [T],
    key: impl Fn(&T) -> K,
) -> BTreeMap<K, Vec<&'a T>> {
    let mut groups: BTreeMap<K, Vec<&'a T>> = BTreeMap::new();
    for row in rows {
        groups.entry(key(row)).or_default().push(row);
    }
    groups
}

/// An evaluated vertex is a candidate for every expected vertex within this distance. The radius is
/// looser than `POSITION_TOLERANCE` so that a vertex a little off is reported as a deviation (its
/// `position_max`) rather than as a missing vertex; the comparison itself still uses
/// `POSITION_TOLERANCE`.
const POSITION_MATCH_RADIUS: f64 = 1.0e-5;
/// Sentinel vertex ID for an evaluated vertex that matches no expected vertex. Rows that use it
/// never equal an expected row, so each such row counts once in the length difference.
const UNMATCHED: usize = usize::MAX;

/// Expected vertex index for each evaluated vertex, or [`UNMATCHED`].
type Matching = Vec<usize>;

/// An expected vertex within [`POSITION_MATCH_RADIUS`] of an evaluated vertex.
struct Candidate {
    expected: usize,
    weight_deviation: f64,
}

/// Edge data that tells coincident vertices apart: seam, sharp, and bevel weight (quantized).
type EdgeFlags = (bool, bool, Option<Ordered>);

fn edge_flags(seam: bool, sharp: bool, weight: Option<f64>) -> EdgeFlags {
    (seam, sharp, weight.map(|weight| Ordered(quantize(weight))))
}

/// Per-mesh topology: sorted neighbours, incident-edge flags, and face-corner counts.
struct Incidence {
    neighbours: Vec<Vec<usize>>,
    flags: Vec<Vec<EdgeFlags>>,
    corners: Vec<usize>,
}

impl Incidence {
    fn new(
        vertex_count: usize,
        edges: impl Iterator<Item = ([usize; 2], EdgeFlags)>,
        corners: impl Iterator<Item = usize>,
    ) -> Self {
        let mut neighbours: Vec<Vec<usize>> = vec![Vec::new(); vertex_count];
        let mut flags: Vec<Vec<EdgeFlags>> = vec![Vec::new(); vertex_count];
        for ([first, second], flag) in edges {
            neighbours[first].push(second);
            neighbours[second].push(first);
            flags[first].push(flag);
            flags[second].push(flag);
        }
        for list in &mut neighbours {
            list.sort_unstable();
        }
        for list in &mut flags {
            list.sort_unstable();
        }
        let mut corner_counts = vec![0_usize; vertex_count];
        for vertex in corners {
            corner_counts[vertex] += 1;
        }
        Self {
            neighbours,
            flags,
            corners: corner_counts,
        }
    }
}

/// Number of entries that two sorted slices share, counting multiplicity.
fn shared_count<T: Ord>(left: &[T], right: &[T]) -> usize {
    let (mut left_index, mut right_index, mut count) = (0, 0, 0);
    while left_index < left.len() && right_index < right.len() {
        match left[left_index].cmp(&right[right_index]) {
            Ordering::Less => left_index += 1,
            Ordering::Greater => right_index += 1,
            Ordering::Equal => {
                count += 1;
                left_index += 1;
                right_index += 1;
            }
        }
    }
    count
}

/// A Weisfeiler–Leman color signature: the previous color prevents merging, while sorted
/// neighbour colors and cyclic face-ring colors refine structural identity without attributes.
#[derive(Clone, Eq, PartialEq, Ord, PartialOrd)]
struct VertexTopologySignature {
    previous: usize,
    neighbours: Vec<usize>,
    faces: Vec<Vec<usize>>,
}

fn canonical_color_ring(colors: &[usize]) -> Vec<usize> {
    let count = colors.len();
    let start = (0..count)
        .min_by(|&left, &right| {
            (0..count)
                .map(|step| colors[(left + step) % count])
                .cmp((0..count).map(|step| colors[(right + step) % count]))
        })
        .unwrap_or(0);
    (0..count)
        .map(|step| colors[(start + step) % count])
        .collect()
}

fn expected_topology_signatures(
    mesh: &CanonMesh,
    incidence: &Incidence,
    colors: &[usize],
) -> Vec<VertexTopologySignature> {
    (0..mesh.vertices.len())
        .map(|vertex| {
            let mut neighbours: Vec<usize> = incidence.neighbours[vertex]
                .iter()
                .map(|&neighbour| colors[neighbour])
                .collect();
            neighbours.sort_unstable();
            let mut faces: Vec<Vec<usize>> = mesh
                .faces
                .iter()
                .filter(|face| face.ring.iter().any(|corner| corner.vertex == vertex))
                .map(|face| {
                    let ring: Vec<usize> = face
                        .ring
                        .iter()
                        .map(|corner| colors[corner.vertex])
                        .collect();
                    canonical_color_ring(&ring)
                })
                .collect();
            faces.sort_unstable();
            VertexTopologySignature {
                previous: colors[vertex],
                neighbours,
                faces,
            }
        })
        .collect()
}

fn actual_topology_signatures(
    mesh: &RawMesh,
    incidence: &Incidence,
    colors: &[usize],
) -> Vec<VertexTopologySignature> {
    (0..mesh.vertices.len())
        .map(|vertex| {
            let mut neighbours: Vec<usize> = incidence.neighbours[vertex]
                .iter()
                .map(|&neighbour| colors[neighbour])
                .collect();
            neighbours.sort_unstable();
            let mut faces: Vec<Vec<usize>> = mesh
                .faces
                .iter()
                .filter(|face| face.ring.iter().any(|corner| corner.vertex == vertex))
                .map(|face| {
                    let ring: Vec<usize> = face
                        .ring
                        .iter()
                        .map(|corner| colors[corner.vertex])
                        .collect();
                    canonical_color_ring(&ring)
                })
                .collect();
            faces.sort_unstable();
            VertexTopologySignature {
                previous: colors[vertex],
                neighbours,
                faces,
            }
        })
        .collect()
}

/// Assigns common initial colors to each connected position-candidate component, then repeatedly
/// refines both meshes together so equal colors mean equal iterated topology signatures.
fn topology_colors(
    expected: &CanonMesh,
    actual: &RawMesh,
    expected_incidence: &Incidence,
    actual_incidence: &Incidence,
    candidates: &[Vec<Candidate>],
) -> TestResult<(Vec<usize>, Vec<usize>)> {
    let uncolored = usize::MAX;
    let mut expected_colors = vec![uncolored; expected.vertices.len()];
    let mut actual_colors = vec![uncolored; actual.vertices.len()];
    let mut actual_for_expected = vec![Vec::new(); expected.vertices.len()];
    for (actual_index, rows) in candidates.iter().enumerate() {
        for candidate in rows {
            actual_for_expected[candidate.expected].push(actual_index);
        }
    }

    let mut next_color = 0;
    for seed in 0..actual.vertices.len() {
        if actual_colors[seed] != uncolored || candidates[seed].is_empty() {
            continue;
        }
        let color = next_color;
        next_color += 1;
        let mut pending = vec![(true, seed)];
        while let Some((is_actual, index)) = pending.pop() {
            if is_actual {
                if actual_colors[index] != uncolored {
                    continue;
                }
                actual_colors[index] = color;
                pending.extend(
                    candidates[index]
                        .iter()
                        .map(|candidate| (false, candidate.expected)),
                );
            } else {
                if expected_colors[index] != uncolored {
                    continue;
                }
                expected_colors[index] = color;
                pending.extend(
                    actual_for_expected[index]
                        .iter()
                        .map(|&actual_index| (true, actual_index)),
                );
            }
        }
    }
    for color in &mut actual_colors {
        if *color == uncolored {
            *color = next_color;
            next_color += 1;
        }
    }
    for color in &mut expected_colors {
        if *color == uncolored {
            *color = next_color;
            next_color += 1;
        }
    }

    loop {
        let expected_signatures =
            expected_topology_signatures(expected, expected_incidence, &expected_colors);
        let actual_signatures =
            actual_topology_signatures(actual, actual_incidence, &actual_colors);
        let mut all_signatures = expected_signatures.clone();
        all_signatures.extend(actual_signatures.iter().cloned());
        all_signatures.sort_unstable();
        all_signatures.dedup();
        let next_expected: Vec<usize> = expected_signatures
            .iter()
            .map(|signature| signature_rank(&all_signatures, signature))
            .collect::<TestResult<_>>()?;
        let next_actual: Vec<usize> = actual_signatures
            .iter()
            .map(|signature| signature_rank(&all_signatures, signature))
            .collect::<TestResult<_>>()?;
        if all_signatures.len() == next_color {
            return Ok((next_expected, next_actual));
        }
        next_color = all_signatures.len();
        expected_colors = next_expected;
        actual_colors = next_actual;
    }
}

/// Position of `signature` in the sorted, deduplicated `signatures`; every signature was collected
/// into that list, so a miss is a broken invariant.
fn signature_rank(
    signatures: &[VertexTopologySignature],
    signature: &VertexTopologySignature,
) -> TestResult<usize> {
    signatures
        .binary_search(signature)
        .map_err(|_| "signature was collected".into())
}

/// Additive lexicographic score for a valid position pair: exact refined color, then local
/// neighbour/face color overlap and incidence gaps, with edge flags and weights last.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
struct AssociationScore([i64; 8]);

impl AssociationScore {
    const ZERO: Self = Self([0; 8]);

    fn plus(self, other: Self) -> Self {
        let mut fields = [0; 8];
        for (index, field) in fields.iter_mut().enumerate() {
            *field = self.0[index] + other.0[index];
        }
        Self(fields)
    }

    fn negated(self) -> Self {
        Self(self.0.map(|field| -field))
    }
}

fn face_color_overlap(expected: &[Vec<usize>], actual: &[Vec<usize>]) -> usize {
    let mut expected_colors: Vec<usize> = expected.iter().flatten().copied().collect();
    let mut actual_colors: Vec<usize> = actual.iter().flatten().copied().collect();
    expected_colors.sort_unstable();
    actual_colors.sort_unstable();
    shared_count(&expected_colors, &actual_colors)
}

fn association_score(
    expected_incidence: &Incidence,
    actual_incidence: &Incidence,
    expected_signature: &VertexTopologySignature,
    actual_signature: &VertexTopologySignature,
    actual_index: usize,
    candidate: &Candidate,
) -> AssociationScore {
    let expected_index = candidate.expected;
    let neighbour_overlap =
        shared_count(&expected_signature.neighbours, &actual_signature.neighbours);
    let face_overlap = face_color_overlap(&expected_signature.faces, &actual_signature.faces);
    let neighbour_gap = expected_signature
        .neighbours
        .len()
        .abs_diff(actual_signature.neighbours.len());
    let face_gap = expected_signature
        .faces
        .len()
        .abs_diff(actual_signature.faces.len());
    let corner_gap =
        expected_incidence.corners[expected_index].abs_diff(actual_incidence.corners[actual_index]);
    let flag_hits = shared_count(
        &expected_incidence.flags[expected_index],
        &actual_incidence.flags[actual_index],
    );
    let weight_penalty = (candidate.weight_deviation / WEIGHT_TOLERANCE).round() as i64;
    AssociationScore([
        i64::from(expected_signature.previous == actual_signature.previous),
        neighbour_overlap as i64,
        face_overlap as i64,
        -(neighbour_gap as i64),
        -(face_gap as i64),
        -(corner_gap as i64),
        flag_hits as i64,
        -weight_penalty,
    ])
}

struct FlowEdge {
    to: usize,
    reverse: usize,
    available: bool,
    score: AssociationScore,
}

fn add_flow_edge(graph: &mut [Vec<FlowEdge>], from: usize, to: usize, score: AssociationScore) {
    let reverse_from = graph[from].len();
    let reverse_to = graph[to].len();
    graph[from].push(FlowEdge {
        to,
        reverse: reverse_to,
        available: true,
        score,
    });
    graph[to].push(FlowEdge {
        to: from,
        reverse: reverse_from,
        available: false,
        score: score.negated(),
    });
}

struct ComponentMatching {
    cardinality: usize,
    score: AssociationScore,
    expected_for_actual: Vec<Option<usize>>,
}

/// Maximum-cardinality matching with lexicographically optimal association score. Successive
/// best augmenting paths can reassign earlier vertices, unlike a storage-order greedy lock.
fn solve_component(
    actuals: &[usize],
    expecteds: &[usize],
    expected_count: usize,
    candidates: &[Vec<(usize, AssociationScore)>],
    forced: Option<(usize, usize)>,
    excluded_actual: Option<usize>,
) -> TestResult<ComponentMatching> {
    let mut expected_for_actual = vec![None; actuals.len()];
    if let Some((actual, expected)) = forced {
        let local_actual = actuals
            .iter()
            .position(|&index| index == actual)
            .ok_or("forced actual is in component")?;
        candidates[actual]
            .iter()
            .find(|(index, _)| *index == expected)
            .ok_or("forced pair is a candidate")?;
        expected_for_actual[local_actual] = Some(expected);
    }
    let remaining_actuals: Vec<usize> = actuals
        .iter()
        .copied()
        .filter(|&actual| {
            Some(actual) != excluded_actual
                && forced.is_none_or(|(forced_actual, _)| forced_actual != actual)
        })
        .collect();
    let remaining_expecteds: Vec<usize> = expecteds
        .iter()
        .copied()
        .filter(|&expected| forced.is_none_or(|(_, forced_expected)| forced_expected != expected))
        .collect();
    let expected_start = 1 + remaining_actuals.len();
    let sink = expected_start + remaining_expecteds.len();
    let mut graph: Vec<Vec<FlowEdge>> = std::iter::repeat_with(Vec::new).take(sink + 1).collect();
    let source = 0;
    let mut local_expected = vec![usize::MAX; expected_count];
    for (local_index, &expected) in remaining_expecteds.iter().enumerate() {
        local_expected[expected] = local_index;
        add_flow_edge(
            &mut graph,
            expected_start + local_index,
            sink,
            AssociationScore::ZERO,
        );
    }
    for (local_index, &actual) in remaining_actuals.iter().enumerate() {
        let actual_node = 1 + local_index;
        add_flow_edge(&mut graph, source, actual_node, AssociationScore::ZERO);
        for &(expected, score) in &candidates[actual] {
            let expected_index = local_expected[expected];
            if expected_index != usize::MAX {
                add_flow_edge(
                    &mut graph,
                    actual_node,
                    expected_start + expected_index,
                    score,
                );
            }
        }
    }

    loop {
        let mut best_score = vec![None; graph.len()];
        let mut previous: Vec<Option<(usize, usize)>> = vec![None; graph.len()];
        best_score[source] = Some(AssociationScore::ZERO);
        for _ in 1..graph.len() {
            let mut changed = false;
            for from in 0..graph.len() {
                let Some(prefix) = best_score[from] else {
                    continue;
                };
                for (edge_index, edge) in graph[from].iter().enumerate() {
                    if !edge.available {
                        continue;
                    }
                    let score = prefix.plus(edge.score);
                    if best_score[edge.to].is_none_or(|current| score > current) {
                        best_score[edge.to] = Some(score);
                        previous[edge.to] = Some((from, edge_index));
                        changed = true;
                    }
                }
            }
            if !changed {
                break;
            }
        }
        if best_score[sink].is_none() {
            break;
        }
        let mut node = sink;
        while node != source {
            let (from, edge_index) =
                previous[node].ok_or("augmenting path predecessor is present")?;
            let reverse = graph[from][edge_index].reverse;
            graph[from][edge_index].available = false;
            graph[node][reverse].available = true;
            node = from;
        }
    }

    for (local_index, &actual) in remaining_actuals.iter().enumerate() {
        let actual_node = 1 + local_index;
        if let Some(edge) = graph[actual_node]
            .iter()
            .find(|edge| !edge.available && (expected_start..sink).contains(&edge.to))
        {
            let local_actual = actuals
                .iter()
                .position(|&index| index == actual)
                .ok_or("remaining actual is in component")?;
            expected_for_actual[local_actual] = Some(remaining_expecteds[edge.to - expected_start]);
        }
    }
    let mut cardinality = 0;
    let mut score = AssociationScore::ZERO;
    for (local_actual, expected) in expected_for_actual.iter().enumerate() {
        let Some(expected) = expected else { continue };
        cardinality += 1;
        let actual = actuals[local_actual];
        score = score.plus(
            candidates[actual]
                .iter()
                .find(|(candidate, _)| candidate == expected)
                .ok_or("matched pair is a candidate")?
                .1,
        );
    }
    Ok(ComponentMatching {
        cardinality,
        score,
        expected_for_actual,
    })
}

fn candidate_components(
    candidates: &[Vec<(usize, AssociationScore)>],
    expected_count: usize,
) -> Vec<(Vec<usize>, Vec<usize>)> {
    let mut actual_for_expected = vec![Vec::new(); expected_count];
    for (actual, rows) in candidates.iter().enumerate() {
        for (expected, _) in rows {
            actual_for_expected[*expected].push(actual);
        }
    }
    let mut seen_actual = vec![false; candidates.len()];
    let mut seen_expected = vec![false; expected_count];
    let mut components = Vec::new();
    for seed in 0..candidates.len() {
        if seen_actual[seed] || candidates[seed].is_empty() {
            continue;
        }
        let mut actuals = Vec::new();
        let mut expecteds = Vec::new();
        let mut pending = vec![(true, seed)];
        while let Some((is_actual, index)) = pending.pop() {
            if is_actual {
                if seen_actual[index] {
                    continue;
                }
                seen_actual[index] = true;
                actuals.push(index);
                pending.extend(
                    candidates[index]
                        .iter()
                        .map(|(expected, _)| (false, *expected)),
                );
            } else {
                if seen_expected[index] {
                    continue;
                }
                seen_expected[index] = true;
                expecteds.push(index);
                pending.extend(
                    actual_for_expected[index]
                        .iter()
                        .map(|&actual| (true, actual)),
                );
            }
        }
        actuals.sort_unstable();
        expecteds.sort_unstable();
        components.push((actuals, expecteds));
    }
    components
}

fn mark_noninterchangeable_changes(
    expected: &CanonMesh,
    actuals: &[usize],
    original: &[Option<usize>],
    alternative: &[Option<usize>],
    ambiguous: &mut [bool],
) {
    for (local_actual, (&before, &after)) in original.iter().zip(alternative).enumerate() {
        if before == after {
            continue;
        }
        let interchangeable = match (before, after) {
            (Some(before), Some(after)) => interchangeable(expected, before, after),
            _ => false,
        };
        if !interchangeable {
            ambiguous[actuals[local_actual]] = true;
        }
    }
}

fn weight_deviation(expected: &[f64], actual: &[f64]) -> f64 {
    expected
        .iter()
        .zip(actual)
        .map(|(expected, actual)| (expected - actual).abs())
        .fold(0.0, f64::max)
}

/// Component-wise `==` with IEEE semantics (`+0.0 == -0.0`, `NaN != NaN`) via `partial_cmp`.
fn components_equal(left: &[f64], right: &[f64]) -> bool {
    left.len() == right.len()
        && left
            .iter()
            .zip(right)
            .all(|(left, right)| left.partial_cmp(right) == Some(Ordering::Equal))
}

/// Two face rows agree when their attributes match and their corners carry the same vertex, UV and
/// normal in the same order.
fn same_face(left: &CanonFace, right: &CanonFace) -> bool {
    left.material == right.material
        && left.smooth == right.smooth
        && left.strength == right.strength
        && left.ring.len() == right.ring.len()
        && left.ring.iter().zip(&right.ring).all(|(left, right)| {
            left.vertex == right.vertex
                && components_equal(&left.uv, &right.uv)
                && left.normal == right.normal
        })
}

/// True when exchanging expected vertices `first` and `second` leaves the expected mesh unchanged:
/// equal data, the same edge rows after relabelling, and the same face rows after the ring
/// normalisation the comparison uses. Then either assignment of the two yields the same comparison.
fn interchangeable(expected: &CanonMesh, first: usize, second: usize) -> bool {
    if expected.vertices[first].key != expected.vertices[second].key {
        return false;
    }
    let swap = |label: usize| {
        if label == first {
            second
        } else if label == second {
            first
        } else {
            label
        }
    };
    let mut swapped_edges: Vec<EdgeOrder> = expected
        .edges
        .iter()
        .map(|edge| {
            let [a, b] = edge.vertices.map(&swap);
            edge_order(&CanonEdge {
                vertices: [a.min(b), a.max(b)],
                seam: edge.seam,
                sharp: edge.sharp,
                weight: edge.weight,
            })
        })
        .collect();
    let mut original_edges: Vec<EdgeOrder> = expected.edges.iter().map(edge_order).collect();
    swapped_edges.sort_unstable();
    original_edges.sort_unstable();
    if swapped_edges != original_edges {
        return false;
    }
    let mut used = vec![false; expected.faces.len()];
    for face in &expected.faces {
        let mut ring: Vec<CanonCorner> = face
            .ring
            .iter()
            .map(|corner| CanonCorner {
                vertex: swap(corner.vertex),
                uv: corner.uv,
                normal: corner.normal,
            })
            .collect();
        rotate_to_minimum(&mut ring, &expected.vertices);
        let swapped = CanonFace {
            material: face.material,
            smooth: face.smooth,
            strength: face.strength,
            ring,
        };
        let Some(index) = expected
            .faces
            .iter()
            .enumerate()
            .position(|(index, original)| !used[index] && same_face(original, &swapped))
        else {
            return false;
        };
        used[index] = true;
    }
    true
}

/// Associates evaluated vertices with expected vertices by maximum-cardinality position matching.
/// Refined topology and local structural signatures outrank edge/group data. Any equally optimal
/// non-interchangeable mapping is reported as ambiguous rather than selected by storage order.
fn match_vertices(
    expected: &CanonMesh,
    actual: &RawMesh,
    report: &mut CaseReport,
) -> TestResult<Matching> {
    let expected_incidence = Incidence::new(
        expected.vertices.len(),
        expected.edges.iter().map(|edge| {
            (
                edge.vertices,
                edge_flags(edge.seam, edge.sharp, edge.weight),
            )
        }),
        expected
            .faces
            .iter()
            .flat_map(|face| face.ring.iter().map(|corner| corner.vertex)),
    );
    let actual_incidence = Incidence::new(
        actual.vertices.len(),
        actual.edges.iter().map(|edge| {
            (
                edge.vertices,
                edge_flags(edge.seam, edge.sharp, edge.weight),
            )
        }),
        actual
            .faces
            .iter()
            .flat_map(|face| face.ring.iter().map(|corner| corner.vertex)),
    );
    let candidates: Vec<Vec<Candidate>> = actual
        .vertices
        .iter()
        .map(|vertex| {
            expected
                .vertices
                .iter()
                .enumerate()
                .filter_map(|(index, candidate)| {
                    let distance = max_abs_difference(candidate.position, vertex.position);
                    (distance <= POSITION_MATCH_RADIUS).then(|| Candidate {
                        expected: index,
                        weight_deviation: weight_deviation(&candidate.weights, &vertex.weights),
                    })
                })
                .collect()
        })
        .collect();
    let (expected_colors, actual_colors) = topology_colors(
        expected,
        actual,
        &expected_incidence,
        &actual_incidence,
        &candidates,
    )?;
    let expected_signatures =
        expected_topology_signatures(expected, &expected_incidence, &expected_colors);
    let actual_signatures = actual_topology_signatures(actual, &actual_incidence, &actual_colors);
    let scored_candidates: Vec<Vec<(usize, AssociationScore)>> = candidates
        .iter()
        .enumerate()
        .map(|(actual_index, rows)| {
            rows.iter()
                .map(|candidate| {
                    (
                        candidate.expected,
                        association_score(
                            &expected_incidence,
                            &actual_incidence,
                            &expected_signatures[candidate.expected],
                            &actual_signatures[actual_index],
                            actual_index,
                            candidate,
                        ),
                    )
                })
                .collect()
        })
        .collect();
    let mut ambiguous = vec![false; actual.vertices.len()];

    let mut matching = vec![UNMATCHED; actual.vertices.len()];
    for (actuals, expecteds) in candidate_components(&scored_candidates, expected.vertices.len()) {
        let best = solve_component(
            &actuals,
            &expecteds,
            expected.vertices.len(),
            &scored_candidates,
            None,
            None,
        )?;
        for (local_actual, expected_index) in best.expected_for_actual.iter().enumerate() {
            if let Some(expected_index) = expected_index {
                matching[actuals[local_actual]] = *expected_index;
            }
        }

        // Test every possible change to an actual row against the component's globally optimal
        // cardinality and score. If an equally optimal mapping changes a non-interchangeable
        // expected identity, retain no association for that actual rather than guessing.
        for &actual_index in &actuals {
            let without_actual = solve_component(
                &actuals,
                &expecteds,
                expected.vertices.len(),
                &scored_candidates,
                None,
                Some(actual_index),
            )?;
            if without_actual.cardinality == best.cardinality && without_actual.score == best.score
            {
                mark_noninterchangeable_changes(
                    expected,
                    &actuals,
                    &best.expected_for_actual,
                    &without_actual.expected_for_actual,
                    &mut ambiguous,
                );
            }
            let local_actual = actuals
                .iter()
                .position(|&index| index == actual_index)
                .ok_or("actual is in component")?;
            for &(expected_index, _) in &scored_candidates[actual_index] {
                if best.expected_for_actual[local_actual] == Some(expected_index) {
                    continue;
                }
                let alternative = solve_component(
                    &actuals,
                    &expecteds,
                    expected.vertices.len(),
                    &scored_candidates,
                    Some((actual_index, expected_index)),
                    None,
                )?;
                if alternative.cardinality == best.cardinality && alternative.score == best.score {
                    mark_noninterchangeable_changes(
                        expected,
                        &actuals,
                        &best.expected_for_actual,
                        &alternative.expected_for_actual,
                        &mut ambiguous,
                    );
                }
            }
        }
    }

    let mut to_actual = vec![None; expected.vertices.len()];
    for (actual_index, expected_index) in matching.iter_mut().enumerate() {
        if ambiguous[actual_index] {
            *expected_index = UNMATCHED;
            report.ambiguous += 1;
        } else if *expected_index != UNMATCHED {
            to_actual[*expected_index] = Some(actual_index);
        }
    }

    let nearest_expected = |vertex: &RawVertex| {
        expected
            .vertices
            .iter()
            .map(|candidate| max_abs_difference(candidate.position, vertex.position))
            .min_by(f64::total_cmp)
            .unwrap_or(f64::INFINITY)
    };
    for (actual_index, vertex) in actual.vertices.iter().enumerate() {
        let expected_index = matching[actual_index];
        if expected_index == UNMATCHED {
            report.position_bad += 1;
            report.position_max = report.position_max.max(nearest_expected(vertex));
            continue;
        }
        let expected_vertex = &expected.vertices[expected_index];
        let distance = max_abs_difference(expected_vertex.position, vertex.position);
        report.position_max = report.position_max.max(distance);
        if distance > POSITION_TOLERANCE {
            report.position_bad += 1;
        }
        for (expected_weight, actual_weight) in expected_vertex.weights.iter().zip(&vertex.weights)
        {
            let deviation = (expected_weight - actual_weight).abs();
            report.weight_max = report.weight_max.max(deviation);
            if deviation > WEIGHT_TOLERANCE {
                report.weight_bad += 1;
            }
        }
    }
    for (expected_index, matched) in to_actual.iter().enumerate() {
        if matched.is_some() {
            continue;
        }
        report.position_bad += 1;
        let expected_position = expected.vertices[expected_index].position;
        let nearest = actual
            .vertices
            .iter()
            .map(|vertex| max_abs_difference(expected_position, vertex.position))
            .min_by(f64::total_cmp)
            .unwrap_or(f64::INFINITY);
        report.position_max = report.position_max.max(nearest);
    }
    Ok(matching)
}

fn compare(name: &str, expected: &CanonMesh, actual: &RawMesh) -> TestResult<CaseReport> {
    let mut report = CaseReport::new(name, expected.vertices.len(), actual.vertices.len());
    let matching = match_vertices(expected, actual, &mut report)?;

    let mut actual_edges = Vec::with_capacity(actual.edges.len());
    for edge in &actual.edges {
        let [first, second] = edge.vertices.map(|vertex| matching[vertex]);
        actual_edges.push(CanonEdge {
            vertices: [first.min(second), first.max(second)],
            seam: edge.seam,
            sharp: edge.sharp,
            weight: edge.weight,
        });
    }
    let expected_edges = group_by(&expected.edges, |edge| edge.vertices);
    let actual_edge_groups = group_by(&actual_edges, |edge| edge.vertices);
    let edge_keys: BTreeSet<[usize; 2]> = expected_edges
        .keys()
        .chain(actual_edge_groups.keys())
        .copied()
        .collect();

    for key in edge_keys {
        let expected_rows = expected_edges.get(&key).map_or(&[][..], Vec::as_slice);
        let actual_rows = actual_edge_groups.get(&key).map_or(&[][..], Vec::as_slice);
        for (expected_edge, actual_edge) in expected_rows.iter().zip(actual_rows) {
            if expected_edge.seam != actual_edge.seam {
                report.flag_bad += 1;
            }
            if expected_edge.sharp != actual_edge.sharp {
                report.flag_bad += 1;
            }
            match (expected_edge.weight, actual_edge.weight) {
                (Some(expected_weight), Some(actual_weight)) => {
                    let deviation = (expected_weight - actual_weight).abs();
                    if deviation > WEIGHT_TOLERANCE {
                        report.edge_bad += 1;
                    }
                }
                (None, None) => {}
                _ => report.edge_bad += 1,
            }
        }
        report.edge_bad += expected_rows.len().abs_diff(actual_rows.len());
    }

    let mut actual_faces = Vec::with_capacity(actual.faces.len());
    for face in &actual.faces {
        let mut ring: Vec<CanonCorner> = face
            .ring
            .iter()
            .map(|corner| CanonCorner {
                vertex: matching[corner.vertex],
                uv: corner.uv,
                normal: corner.normal,
            })
            .collect();
        if ring.iter().all(|corner| corner.vertex != UNMATCHED) {
            rotate_to_minimum(&mut ring, &expected.vertices);
        }
        actual_faces.push(CanonFace {
            material: face.material,
            smooth: face.smooth,
            strength: face.strength,
            ring,
        });
    }
    let expected_faces = group_by(&expected.faces, |face| {
        face.ring
            .iter()
            .map(|corner| corner.vertex)
            .collect::<Vec<usize>>()
    });
    let actual_face_groups = group_by(&actual_faces, |face| {
        face.ring
            .iter()
            .map(|corner| corner.vertex)
            .collect::<Vec<usize>>()
    });
    let ring_keys: BTreeSet<Vec<usize>> = expected_faces
        .keys()
        .chain(actual_face_groups.keys())
        .cloned()
        .collect();
    for key in ring_keys {
        let expected_rows = expected_faces.get(&key).map_or(&[][..], Vec::as_slice);
        let actual_rows = actual_face_groups.get(&key).map_or(&[][..], Vec::as_slice);
        for (expected_face, actual_face) in expected_rows.iter().zip(actual_rows) {
            if expected_face.material != actual_face.material {
                report.flag_bad += 1;
            }
            if expected_face.smooth != actual_face.smooth {
                report.flag_bad += 1;
            }
            if expected_face.strength != actual_face.strength {
                report.flag_bad += 1;
            }
            for (expected_corner, actual_corner) in expected_face.ring.iter().zip(&actual_face.ring)
            {
                for (expected_uv, actual_uv) in expected_corner.uv.iter().zip(actual_corner.uv) {
                    let deviation = (expected_uv - actual_uv).abs();
                    report.uv_max = report.uv_max.max(deviation);
                    if deviation > UV_TOLERANCE {
                        report.uv_bad += 1;
                    }
                }
                let normal = max_abs_difference(expected_corner.normal, actual_corner.normal);
                report.normal_max = report.normal_max.max(normal);
                if normal > NORMAL_TOLERANCE {
                    report.normal_bad += 1;
                }
            }
        }
        report.ring_bad += expected_rows.len().abs_diff(actual_rows.len());
    }
    Ok(report)
}

fn print_table(reports: &[CaseReport]) {
    println!(
        "bevel oracle: {} cases; tolerances: position/weight/uv {POSITION_TOLERANCE:e}, normal {NORMAL_TOLERANCE}",
        reports.len()
    );
    println!(
        "{:<52} {:>9} {:>7} {:>10} {:>7} {:>10} {:>7} {:>10} {:>7} {:>10} {:>5} {:>5} {:>5}  status",
        "case",
        "verts e/a",
        "pos_bad",
        "pos_max",
        "wgt_bad",
        "wgt_max",
        "uv_bad",
        "uv_max",
        "nrm_bad",
        "nrm_max",
        "ring",
        "edge",
        "flag"
    );
    for report in reports {
        if let Some(error) = &report.error {
            println!("{:<52} ERROR {error}", report.name);
            continue;
        }
        let status = match (report.failed(), report.ambiguous) {
            (false, _) => "ok".to_owned(),
            (true, 0) => "FAIL".to_owned(),
            (true, count) => format!("FAIL ambiguous={count}"),
        };
        println!(
            "{:<52} {:>4}/{:<4} {:>7} {:>10.3e} {:>7} {:>10.3e} {:>7} {:>10.3e} {:>7} {:>10.3e} {:>5} {:>5} {:>5}  {status}",
            report.name,
            report.expected_vertices,
            report.actual_vertices,
            report.position_bad,
            report.position_max,
            report.weight_bad,
            report.weight_max,
            report.uv_bad,
            report.uv_max,
            report.normal_bad,
            report.normal_max,
            report.ring_bad,
            report.edge_bad,
            report.flag_bad,
        );
    }
}

// ---------------------------------------------------------------------------------------------
// Oracle input: the mesh Blender started from, expressed in Potter's attribute layout.
// ---------------------------------------------------------------------------------------------

fn input_mesh(input: &Value) -> TestResult<Mesh> {
    let positions = array(&input["vertices"], "oracle vertices must be an array")?
        .iter()
        .map(|point| Ok(DVec3::from_array(bit_triple(point, "oracle position")?)))
        .collect::<TestResult<Vec<_>>>()?;
    let polygons = array(&input["faces"], "oracle faces must be an array")?
        .iter()
        .map(|face| {
            array(face, "oracle face must be an array")?
                .iter()
                .map(|index| index_value(index, "face vertex index must be unsigned"))
                .collect::<TestResult<Vec<usize>>>()
        })
        .collect::<TestResult<Vec<_>>>()?;
    let input_edges = array(&input["edges"], "oracle edges must be an array")?;
    let mut mesh = Mesh::new();
    for position in positions {
        mesh.insert_vertex(position)?;
    }
    // Blender's BMesh import inserts `Mesh.edges` in order, with each stored endpoint direction.
    for edge in input_edges {
        let endpoints = array(&edge["vertices"], "oracle edge vertices must be an array")?;
        let [first, second] = endpoints.as_slice() else {
            return Err("oracle edge must have two endpoints".into());
        };
        mesh.insert_edge([
            u32::try_from(index_value(first, "edge endpoint must be unsigned")?)?,
            u32::try_from(index_value(second, "edge endpoint must be unsigned")?)?,
        ])?;
    }
    for polygon in polygons {
        let vertices = polygon
            .into_iter()
            .map(u32::try_from)
            .collect::<Result<Vec<_>, _>>()?;
        mesh.insert_face(vertices, 0)?;
    }

    let materials = array(
        &input["face_material_indices"],
        "oracle material indices must be an array",
    )?;
    let smooth = array(
        &input["face_smooth"],
        "oracle face smooth flags must be an array",
    )?;
    if materials.len() != mesh.faces.len() || smooth.len() != mesh.faces.len() {
        return Err("oracle per-face arrays differ from the face count".into());
    }
    let mut smooth_values = Map::new();
    for ((face, material), flag) in mesh.faces.iter_mut().zip(materials).zip(smooth) {
        face.material_index =
            u32::try_from(material.as_u64().ok_or("material index must be unsigned")?)?;
        smooth_values.insert(
            format!("f{}", face.id),
            json!(flag.as_bool().ok_or("face smooth flag must be boolean")?),
        );
    }
    mesh.attributes.insert(
        "shade_smooth".to_owned(),
        json!({"domain":"face","type":"bool","values":smooth_values}),
    );

    let group_names = array(
        &input["vertex_groups"]["names"],
        "oracle vertex group names must be an array",
    )?;
    let group_weights = array(
        &input["vertex_groups"]["weights"],
        "oracle vertex group weights must be an array",
    )?;
    if group_weights.len() != mesh.vertices.len() {
        return Err("oracle vertex group rows differ from vertex count".into());
    }
    let mut groups = Map::new();
    for (group_index, group_name) in group_names.iter().enumerate() {
        let name = group_name
            .as_str()
            .ok_or("vertex group name must be a string")?;
        let mut weights = Map::new();
        for (vertex, row) in mesh.vertices.iter().zip(group_weights) {
            let bits = row
                .as_array()
                .and_then(|values| values.get(group_index))
                .ok_or("oracle vertex group weight is missing")?;
            weights.insert(format!("v{}", vertex.id), json!(float_bits(bits)?));
        }
        groups.insert(name.to_owned(), Value::Object(weights));
    }
    mesh.attributes
        .insert("vertex_groups".to_owned(), Value::Object(groups));

    let mut edge_rows = HashMap::new();
    for edge in input_edges {
        let endpoints = array(&edge["vertices"], "oracle edge vertices must be an array")?;
        let [first, second] = endpoints.as_slice() else {
            return Err("oracle edge must have two endpoints".into());
        };
        let first = index_value(first, "edge endpoint must be unsigned")?;
        let second = index_value(second, "edge endpoint must be unsigned")?;
        edge_rows.insert(edge_key(first, second), edge);
    }
    if edge_rows.len() != mesh.edges.len() {
        return Err("oracle edge set differs from mesh face boundaries".into());
    }
    let mut sharp_values = Map::new();
    let mut seam_values = Map::new();
    let mut edge_weights = Map::new();
    for edge in &mesh.edges {
        let source = edge_rows
            .get(&edge_key(
                usize::try_from(edge.vertices[0])?,
                usize::try_from(edge.vertices[1])?,
            ))
            .ok_or("mesh edge is missing from oracle input")?;
        let key = format!("e{}", edge.id);
        let seam = source["seam"]
            .as_bool()
            .ok_or("edge seam flag must be boolean")?;
        let sharp = source["sharp"]
            .as_bool()
            .ok_or("edge sharp flag must be boolean")?;
        seam_values.insert(key.clone(), json!(seam));
        sharp_values.insert(key.clone(), json!(sharp));
        edge_weights.insert(key, json!(float_bits(&source["bevel_weight"])?));
    }
    mesh.attributes.insert(
        "seam".to_owned(),
        json!({"domain":"edge","type":"bool","values":seam_values}),
    );
    mesh.attributes.insert(
        "sharp".to_owned(),
        json!({"domain":"edge","type":"bool","values":sharp_values}),
    );
    mesh.attributes.insert(
        "bevel_weight_edge".to_owned(),
        json!({"domain":"edge","type":"float","values":edge_weights}),
    );

    let vertex_weight_values = array(
        &input["vertex_bevel_weight"],
        "oracle vertex bevel weights must be an array",
    )?;
    if vertex_weight_values.len() != mesh.vertices.len() {
        return Err("oracle vertex bevel weights differ from vertex count".into());
    }
    let mut vertex_weights = Map::new();
    for (vertex, value) in mesh.vertices.iter().zip(vertex_weight_values) {
        vertex_weights.insert(format!("v{}", vertex.id), json!(float_bits(value)?));
    }
    mesh.attributes.insert(
        "bevel_weight_vert".to_owned(),
        json!({"domain":"point","type":"float","values":vertex_weights}),
    );

    let uv_values = array(
        &input["uv_layers"]["UVMap"],
        "oracle UVMap values must be an array",
    )?;
    let corner_count: usize = mesh.faces.iter().map(|face| face.vertices.len()).sum();
    if uv_values.len() != corner_count {
        return Err("oracle UV count differs from mesh corner count".into());
    }
    let uv_pins = input
        .get("uv_pins")
        .filter(|pins| !pins.is_null())
        .map(|pins| {
            let pins = array(pins, "oracle UV pins must be an array")?;
            if pins.len() != corner_count {
                return Err("oracle UV pin count differs from mesh corner count".into());
            }
            pins.iter()
                .map(|pin| {
                    pin.as_bool()
                        .ok_or_else(|| "oracle UV pin must be boolean".into())
                })
                .collect::<TestResult<Vec<_>>>()
        })
        .transpose()?
        .filter(|pins| pins.iter().any(|pinned| *pinned));
    let mut uv_faces = Vec::with_capacity(mesh.faces.len());
    let mut corner = 0;
    for face in &mesh.faces {
        let end = corner + face.vertices.len();
        let values = uv_values[corner..end]
            .iter()
            .map(|value| {
                let [u, v] = bit_pair(value, "oracle UV values")?;
                Ok(json!([u, 1.0 - v]))
            })
            .collect::<TestResult<Vec<_>>>()?;
        let mut row = json!({"face_id": face.id, "uv": values});
        if let Some(pins) = &uv_pins {
            row["pinned"] = json!(&pins[corner..end]);
        }
        uv_faces.push(row);
        corner = end;
    }
    mesh.attributes.insert(
        "uv_layers".to_owned(),
        json!({
            "layers": [{"name": "UVMap", "faces": uv_faces}],
            "active": "UVMap",
            "render": "UVMap"
        }),
    );

    if !input["custom_normal_codes"].is_null() {
        let custom_normal_codes = array(
            &input["custom_normal_codes"],
            "oracle custom normal codes must be an array",
        )?;
        if custom_normal_codes.len() != corner_count {
            return Err("oracle custom normal code count differs from mesh corner count".into());
        }
        // Blender's captured loops are in polygon/corner order, which this input mesh preserves.
        // The canonical identity basis is therefore implicit; codes remain the authoritative state.
        let mut rows = Map::new();
        let mut corner = 0;
        for face in &mesh.faces {
            let end = corner + face.vertices.len();
            let row = custom_normal_codes[corner..end]
                .iter()
                .map(|raw| {
                    let pair = array(raw, "oracle custom normal code must be an array")?;
                    let [first, second] = pair.as_slice() else {
                        return Err("oracle custom normal code must have two components".into());
                    };
                    let pair = [
                        i16::try_from(
                            first
                                .as_i64()
                                .ok_or("normal code component must be an integer")?,
                        )?,
                        i16::try_from(
                            second
                                .as_i64()
                                .ok_or("normal code component must be an integer")?,
                        )?,
                    ];
                    Ok(if pair == [0, 0] {
                        Value::Null
                    } else {
                        json!(pair)
                    })
                })
                .collect::<TestResult<Vec<Value>>>()?;
            rows.insert(format!("f{}", face.id), Value::Array(row));
            corner = end;
        }
        // `values` presence allocates Blender's custom layer, including all-Auto inputs.
        mesh.attributes.insert(
            "custom_normal".to_owned(),
            json!({"domain":"corner","type":"normal_code","values":rows}),
        );

        // Keep the captured code state exact and face/corner aligned before production validation.
        let stored = mesh.attributes["custom_normal"]["values"]
            .as_object()
            .ok_or("native custom-normal values must be face-keyed")?;
        let mut corner = 0;
        for face in &mesh.faces {
            let end = corner + face.vertices.len();
            let row = stored
                .get(&format!("f{}", face.id))
                .and_then(Value::as_array)
                .filter(|row| row.len() == face.vertices.len())
                .ok_or("native custom-normal row must match its face corners")?;
            for (value, raw) in row.iter().zip(&custom_normal_codes[corner..end]) {
                let pair = array(raw, "oracle custom normal code must be an array")?;
                let [first, second] = pair.as_slice() else {
                    return Err("oracle custom normal code must have two components".into());
                };
                let auto = first.as_i64() == Some(0) && second.as_i64() == Some(0);
                if (auto && !value.is_null()) || (!auto && value.as_array() != Some(pair)) {
                    return Err(format!(
                        "captured custom-normal code state changed at f{} corner {}",
                        face.id, corner
                    )
                    .into());
                }
                corner += 1;
            }
        }
    }
    mesh.validate()?;
    Ok(mesh)
}

fn edge_key(first: usize, second: usize) -> (usize, usize) {
    if first <= second {
        (first, second)
    } else {
        (second, first)
    }
}

// ---------------------------------------------------------------------------------------------
// UV-layer evaluation on a unit cube.
// ---------------------------------------------------------------------------------------------

/// The unit cube of the UV-layer tests: vertex positions and quad faces.
const CUBE_VERTICES: [[f64; 3]; 8] = [
    [0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [1.0, 1.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
    [1.0, 0.0, 1.0],
    [1.0, 1.0, 1.0],
    [0.0, 1.0, 1.0],
];
const CUBE_FACES: [[usize; 4]; 6] = [
    [0, 3, 2, 1],
    [4, 5, 6, 7],
    [0, 1, 5, 4],
    [3, 7, 6, 2],
    [0, 4, 7, 3],
    [1, 2, 6, 5],
];

fn modifier(params: &Value) -> TestResult<Modifier> {
    Ok(Modifier {
        id: Id::new("bevel_uv")?,
        modifier_type: "bevel".to_owned(),
        name: "bevel".to_owned(),
        enabled: true,
        show_viewport: true,
        show_render: true,
        params: params
            .as_object()
            .ok_or("bevel params must be an object")?
            .clone(),
        binding_data: None,
        runtime: ModifierRuntime::default(),
    })
}

fn evaluate_bevel(mesh: &Mesh, params: &Value) -> TestResult<Result<Mesh, PotError>> {
    Ok(evaluate_modifiers(mesh, &[modifier(params)?], "TestMesh"))
}

/// A unit cube with one UV layer, `UVMap`, whose corner values depend only on the corner's vertex.
fn uv_cube() -> TestResult<Mesh> {
    let positions = CUBE_VERTICES
        .iter()
        .map(|&[x, y, z]| DVec3::new(x, y, z))
        .collect();
    let faces = CUBE_FACES.iter().map(|face| face.to_vec()).collect();
    let mut mesh = Mesh::from_positions_and_faces(positions, faces)?;
    let values: Vec<Value> = CUBE_FACES
        .iter()
        .flatten()
        .map(|&vertex| {
            let [x, y, z] = CUBE_VERTICES[vertex];
            json!([0.2 + 0.5 * x + 0.1 * y, 0.3 + 0.6 * z])
        })
        .collect();
    let faces: Vec<Value> = mesh
        .faces
        .iter()
        .zip(values.chunks(4))
        .map(|(face, uv)| json!({"face_id": face.id, "uv": uv}))
        .collect();
    mesh.attributes.insert(
        "uv_layers".to_owned(),
        json!({"layers": [{"name": "UVMap", "faces": faces}], "active": "UVMap", "render": "UVMap"}),
    );
    Ok(mesh)
}

/// Asserts that `mesh` keeps `UVMap` with one row per face and one UV per face corner.
fn assert_uv_map_covers_corners(mesh: &Mesh) -> TestResult {
    let layer = mesh
        .attributes
        .get("uv_layers")
        .and_then(|store| store["layers"].as_array())
        .and_then(|layers| layers.iter().find(|layer| layer["name"] == "UVMap"))
        .ok_or("evaluated mesh lost UVMap")?;
    let rows = array(&layer["faces"], "evaluated UVMap faces must be an array")?;
    for face in &mesh.faces {
        let row = rows
            .iter()
            .find(|row| row["face_id"].as_u64() == Some(u64::from(face.id)))
            .ok_or("evaluated UVMap is missing a face")?;
        assert_eq!(
            row["uv"].as_array().map(Vec::len),
            Some(face.vertices.len()),
            "UV corners of face {}",
            face.id
        );
    }
    Ok(())
}

#[test]
fn edges_bevel_with_uv_layers_partial_edge_selection_evaluates() -> TestResult {
    let mut mesh = uv_cube()?;
    // Only the bottom face's vertices are in the group, so its four edges are selected and the
    // vertical edges are not: each bottom corner has three edges, of which two are beveled.
    mesh.attributes.insert(
        "vertex_groups".to_owned(),
        json!({"bottom": {
            "v0": 1.0, "v1": 1.0, "v2": 1.0, "v3": 1.0, "v4": 0.0, "v5": 0.0, "v6": 0.0, "v7": 0.0
        }}),
    );
    let params = json!({
        "width": 0.1,
        "segments": 3,
        "limit_method": "VGROUP",
        "vertex_group": "bottom",
    });
    let output = evaluate_bevel(&mesh, &params)??;
    assert!(
        output.vertices.len() > mesh.vertices.len(),
        "the selected edges must be beveled"
    );
    assert_uv_map_covers_corners(&output)
}

#[test]
fn edges_bevel_with_uv_layers_profile_zero_evaluates() -> TestResult {
    let mesh = uv_cube()?;
    for segments in [2, 3] {
        let params =
            json!({"width": 0.1, "segments": segments, "profile": 0.0, "limit_method": "NONE"});
        let output = evaluate_bevel(&mesh, &params)??;
        assert!(
            output.vertices.len() > mesh.vertices.len(),
            "profile 0 must bevel (segments {segments})"
        );
        assert_uv_map_covers_corners(&output)?;
    }
    Ok(())
}

#[test]
fn square_profile_even_segments_bevel_uv_cube_matches_blender_counts() -> TestResult {
    let mesh = uv_cube()?;
    // Vertex and face counts of Blender 5.2.2 for the same bevel (profile 1.0, width 0.1).
    for (segments, vertices, faces) in [(2, 56, 54), (3, 96, 98), (4, 152, 150)] {
        let params = json!({
            "width": 0.1,
            "segments": segments,
            "profile": 1.0,
            "limit_method": "NONE",
        });
        let output = evaluate_bevel(&mesh, &params)??;
        assert_eq!(
            output.vertices.len(),
            vertices,
            "vertices at segments {segments}"
        );
        assert_eq!(output.faces.len(), faces, "faces at segments {segments}");
        assert_uv_map_covers_corners(&output)?;
    }
    Ok(())
}
