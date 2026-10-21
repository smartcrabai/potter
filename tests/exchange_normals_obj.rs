//! OBJ shading normals: `vn` and `f v//vn` export, `s` smooth groups, and import of normals.

#[path = "common/exchange_normals.rs"]
#[expect(
    dead_code,
    reason = "each test binary uses a subset of the shared fixture helpers"
)]
mod exchange_normals;

use std::{
    collections::BTreeSet,
    fs,
    path::{Path, PathBuf},
    process::Output,
};

use exchange_normals::{
    NORMAL_TOLERANCE, ObservedCorner, TestResult, assert_matches_reference, export, export_strict,
    init, near, obj_to_potter, observed_corners, pot, write_fixture_scene,
};
use serde_json::{Value, json};
use tempfile::tempdir;

/// One `o` object of a parsed OBJ file, with face indices and corner normals as written.
struct ObjObject {
    name: String,
    vn_lines: Vec<String>,
    shading: Vec<String>,
    faces: Vec<Vec<usize>>,
    corner_normals: Vec<[f64; 3]>,
}

/// Potter-space geometry of an OBJ file. Potter writes OBJ with Blender's default axes, so every
/// position and normal is converted back with `obj_to_potter` as it is parsed.
struct ObjFile {
    positions: Vec<[f64; 3]>,
    objects: Vec<ObjObject>,
}

fn parse_vec3<'a>(mut words: impl Iterator<Item = &'a str>) -> TestResult<[f64; 3]> {
    let mut value = [0.0_f64; 3];
    for slot in &mut value {
        *slot = words
            .next()
            .ok_or("OBJ vector is missing a component")?
            .parse::<f64>()?;
    }
    Ok(value)
}

fn parse_index(text: &str) -> TestResult<usize> {
    Ok(text
        .parse::<usize>()?
        .checked_sub(1)
        .ok_or("OBJ index is zero")?)
}

fn current(objects: &mut [ObjObject]) -> TestResult<&mut ObjObject> {
    Ok(objects
        .last_mut()
        .ok_or("OBJ line appears before any object")?)
}

/// Parses the parts of an OBJ file the normals tests inspect. Every face corner must carry a
/// normal index, as `v//vn` requires.
fn parse_obj(text: &str) -> TestResult<ObjFile> {
    let mut positions = Vec::new();
    let mut normals = Vec::new();
    let mut objects = Vec::<ObjObject>::new();
    for line in text.lines() {
        let mut words = line.split_whitespace();
        let Some(kind) = words.next() else {
            continue;
        };
        match kind {
            "o" => objects.push(ObjObject {
                name: words.collect::<Vec<_>>().join(" "),
                vn_lines: Vec::new(),
                shading: Vec::new(),
                faces: Vec::new(),
                corner_normals: Vec::new(),
            }),
            "v" => positions.push(obj_to_potter(parse_vec3(words)?)),
            "vn" => {
                normals.push(obj_to_potter(parse_vec3(words)?));
                current(&mut objects)?.vn_lines.push(line.to_owned());
            }
            "s" => current(&mut objects)?.shading.push(line.to_owned()),
            "f" => {
                let mut face = Vec::new();
                let mut corners = Vec::new();
                for token in words {
                    let mut parts = token.split('/');
                    let vertex =
                        parse_index(parts.next().ok_or("face corner has no vertex index")?)?;
                    parts.next();
                    let normal = parse_index(
                        parts
                            .next()
                            .filter(|value| !value.is_empty())
                            .ok_or("face corner has no normal index (expected v//vn)")?,
                    )?;
                    face.push(vertex);
                    corners.push(*normals.get(normal).ok_or("normal index out of range")?);
                }
                let object = current(&mut objects)?;
                object.faces.push(face);
                object.corner_normals.extend(corners);
            }
            _ => {}
        }
    }
    Ok(ObjFile { positions, objects })
}

/// Every face corner of every object, in world space.
fn corners(file: &ObjFile) -> Vec<ObservedCorner> {
    file.objects
        .iter()
        .flat_map(|object| observed_corners(&file.positions, &object.faces, &object.corner_normals))
        .collect()
}

fn close(left: [f64; 3], right: [f64; 3]) -> bool {
    (0..3).all(|axis| (left[axis] - right[axis]).abs() <= NORMAL_TOLERANCE)
}

fn output_json(output: &Output) -> TestResult<Value> {
    Ok(serde_json::from_slice(&output.stdout)?)
}

/// Feature ids of the `losses` array in a CLI result or error details object.
fn loss_features(value: &Value) -> Vec<&str> {
    value["losses"]
        .as_array()
        .map(|losses| {
            losses
                .iter()
                .filter_map(|loss| loss["feature_id"].as_str())
                .collect()
        })
        .unwrap_or_default()
}

fn write_obj(directory: &Path, name: &str, text: &str) -> TestResult<PathBuf> {
    let path = directory.join(name);
    fs::write(&path, text)?;
    Ok(path)
}

fn import_obj(scene: &Path, file: &Path, allow_lossy: bool) -> TestResult<Output> {
    let mut command = pot();
    command
        .arg("import")
        .arg(scene)
        .arg("--file")
        .arg(file)
        .args([
            "--format",
            "obj",
            "--base-revision",
            "0",
            "--mode",
            "replace",
        ]);
    if allow_lossy {
        command.arg("--allow-lossy");
    }
    Ok(command.arg("--json").output()?)
}

#[test]
fn obj_export_matches_blender_reference_corners() -> TestResult {
    let directory = tempdir()?;
    let scene = directory.path().join("scene");
    write_fixture_scene(&scene)?;
    let out = directory.path().join("scene.obj");
    let output = export_strict(&scene, "obj", &out)?;
    assert!(
        output.status.success(),
        "OBJ export reported losses: {}",
        String::from_utf8_lossy(&output.stdout)
    );

    let file = parse_obj(&fs::read_to_string(&out)?)?;
    let mut names = file
        .objects
        .iter()
        .map(|object| object.name.as_str())
        .collect::<Vec<_>>();
    names.sort_unstable();
    assert_eq!(names, ["mirror", "mixed", "nonuni"]);
    for object in &file.objects {
        assert_eq!(object.faces.len(), 6, "{}", object.name);
        assert_eq!(object.corner_normals.len(), 24, "{}", object.name);
    }
    let observed = corners(&file);
    assert_eq!(observed.len(), 72);
    assert_matches_reference(&observed)?;

    // A mirrored node's faces are reversed, so each face's winding normal agrees with its vn.
    // Face 2 carries a deliberately tilted custom normal, so it is left out.
    let mirror = file
        .objects
        .iter()
        .find(|object| object.name == "mirror")
        .ok_or("mirror object is missing")?;
    // f3 (-Y) carries custom normals, which need not follow its winding; every other face must.
    for (index, face) in mirror
        .faces
        .iter()
        .enumerate()
        .filter(|(index, _)| *index != 3)
    {
        let points = face
            .iter()
            .map(|vertex| file.positions[*vertex])
            .collect::<Vec<_>>();
        let winding = newell_normal(&points);
        let normal = mirror.corner_normals[index * 4];
        let dot = (0..3).map(|axis| winding[axis] * normal[axis]).sum::<f64>();
        assert!(dot > 0.0, "mirror face {index} winds against its normals");
    }
    Ok(())
}

/// Newell normal of a polygon, unnormalized: its direction follows the winding.
fn newell_normal(points: &[[f64; 3]]) -> [f64; 3] {
    let mut area = [0.0_f64; 3];
    for (index, current) in points.iter().enumerate() {
        let next = points[(index + 1) % points.len()];
        area[0] += (current[1] - next[1]) * (current[2] + next[2]);
        area[1] += (current[2] - next[2]) * (current[0] + next[0]);
        area[2] += (current[0] - next[0]) * (current[1] + next[1]);
    }
    area
}

#[test]
fn obj_export_writes_shared_normals_and_smooth_groups() -> TestResult {
    let directory = tempdir()?;
    let scene = directory.path().join("scene");
    write_fixture_scene(&scene)?;
    let out = export(&scene, "obj", &directory.path().join("scene.obj"))?;
    let file = parse_obj(&fs::read_to_string(&out)?)?;

    for object in &file.objects {
        let unique = object.vn_lines.iter().collect::<BTreeSet<_>>();
        assert_eq!(
            unique.len(),
            object.vn_lines.len(),
            "{} repeats a vn line",
            object.name
        );
    }

    let mixed = file
        .objects
        .iter()
        .find(|object| object.name == "mixed")
        .ok_or("OBJ has no object named mixed")?;
    let shading = mixed.shading.iter().map(String::as_str).collect::<Vec<_>>();
    // Smooth-group runs in face order: f0..f3 smooth, f4 flat (-Z), f5 smooth.
    assert_eq!(shading, ["s 1", "s 0", "s 1"]);

    // f4 is flat, the -Z face of Blender's cube order: its four corners all carry (0, 0, -1).
    for normal in &mixed.corner_normals[4 * 4..4 * 4 + 4] {
        assert!(close(*normal, [0.0, 0.0, -1.0]), "f4 corner {normal:?}");
    }
    // f0 = [0, 1, 3, 2] shares vertex 0 with f4 = [2, 6, 4, 0] across the sharp edge e0 = (v0, v2),
    // so its corner at vertex 0 (corner 0) must carry a different vn value than the flat -Z face.
    assert_eq!(mixed.faces[0][0], mixed.faces[4][3]);
    let across_edge = mixed.corner_normals[0];
    assert!(
        !close(across_edge, [0.0, 0.0, -1.0]),
        "sharp edge e0 should split vertex 0 normals, got {across_edge:?} at f0"
    );
    Ok(())
}

#[test]
fn obj_round_trip_through_import_keeps_corner_normals() -> TestResult {
    let directory = tempdir()?;
    let scene = directory.path().join("scene");
    write_fixture_scene(&scene)?;
    let original_path = export(&scene, "obj", &directory.path().join("scene.obj"))?;
    let original = corners(&parse_obj(&fs::read_to_string(&original_path)?)?);
    assert_eq!(original.len(), 72);

    let imported = directory.path().join("imported");
    init(&imported)?;
    let output = pot()
        .arg("import")
        .arg(&imported)
        .arg("--file")
        .arg(&original_path)
        .args([
            "--format",
            "obj",
            "--base-revision",
            "0",
            "--mode",
            "replace",
            "--json",
        ])
        .output()?;
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stdout)
    );
    let result = output_json(&output)?;
    assert!(
        loss_features(&result["result"]).is_empty(),
        "round trip reported losses: {:?}",
        loss_features(&result["result"])
    );

    let again = export(&imported, "obj", &directory.path().join("again.obj"))?;
    let reexported = corners(&parse_obj(&fs::read_to_string(&again)?)?);
    assert_eq!(reexported.len(), original.len());
    for expected in &original {
        let found = reexported
            .iter()
            .filter(|corner| {
                near(corner.corner, expected.corner)
                    && near(corner.face_center, expected.face_center)
            })
            .collect::<Vec<_>>();
        assert_eq!(
            found.len(),
            1,
            "expected one re-exported corner at {:?}",
            expected.corner
        );
        assert!(
            close(found[0].normal, expected.normal),
            "re-exported normal {:?} differs from original {:?} at {:?}",
            found[0].normal,
            expected.normal,
            expected.corner
        );
    }
    assert_matches_reference(&reexported)?;
    Ok(())
}

#[test]
fn obj_import_without_normals_or_smooth_groups_reports_no_losses() -> TestResult {
    let directory = tempdir()?;
    let scene = directory.path().join("scene");
    init(&scene)?;
    let source = write_obj(
        directory.path(),
        "plain.obj",
        "o Plain\nv 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n",
    )?;
    let output = import_obj(&scene, &source, false)?;
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stdout)
    );
    let result = output_json(&output)?["result"].clone();
    assert!(
        loss_features(&result).is_empty(),
        "unexpected losses: {:?}",
        loss_features(&result)
    );
    Ok(())
}

#[test]
fn obj_import_partial_vertex_normals_reports_mesh_normals_loss() -> TestResult {
    let directory = tempdir()?;
    let scene = directory.path().join("scene");
    init(&scene)?;
    let source = write_obj(
        directory.path(),
        "partial.obj",
        "o Partial\nv 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nvn 0 0 1\nf 1//1 2 3 4\n",
    )?;

    let rejected = import_obj(&scene, &source, false)?;
    assert!(
        !rejected.status.success(),
        "partial normals imported without --allow-lossy"
    );
    let rejection = output_json(&rejected)?;
    assert_eq!(rejection["error"]["code"], "UNREPRESENTABLE_FEATURE");
    assert!(
        loss_features(&rejection["error"]["details"]["result"]).contains(&"mesh.normals"),
        "{rejection}"
    );

    let accepted = import_obj(&scene, &source, true)?;
    assert!(
        accepted.status.success(),
        "{}",
        String::from_utf8_lossy(&accepted.stdout)
    );
    let result = output_json(&accepted)?["result"].clone();
    assert_eq!(result["committed"], true);
    assert!(loss_features(&result).contains(&"mesh.normals"), "{result}");
    Ok(())
}

/// The `shade_smooth` values of the imported mesh, one per face, from the scene file.
fn shade_smooth_values(scene: &Path) -> TestResult<Vec<Value>> {
    let document: Value = serde_json::from_slice(&fs::read(scene.join("scene.json"))?)?;
    let blocks = document["data_blocks"]
        .as_object()
        .ok_or("scene.json has no data_blocks")?;
    let mut values = Vec::new();
    for block in blocks.values() {
        if let Some(shading) = block["mesh"]["attributes"]["shade_smooth"]["values"].as_object() {
            values.extend(shading.values().cloned());
        }
    }
    Ok(values)
}

/// Blender's `set_face_sharpness` (`obj_import_mesh.cc`) without `vn`: `sharp_face = !shaded_smooth`, so
/// the `s 1` face is smooth and the `s off` face is flat, with no loss. Probed in Blender 5.2.2, which
/// gives `use_smooth` [true, false] and `sharp_face` [false, true].
#[test]
fn obj_import_smooth_groups_without_normals_sets_sharp_face_per_face() -> TestResult {
    let directory = tempdir()?;
    let scene = directory.path().join("scene");
    init(&scene)?;
    let source = write_obj(
        directory.path(),
        "smooth.obj",
        "o Smooth\nv 0 0 0\nv 1 0 0\nv 0 1 0\nv 1 1 0\ns 1\nf 1 2 3\ns off\nf 2 4 3\n",
    )?;
    let imported = import_obj(&scene, &source, false)?;
    assert!(
        imported.status.success(),
        "{}",
        String::from_utf8_lossy(&imported.stdout)
    );
    let result = output_json(&imported)?["result"].clone();
    assert_eq!(loss_features(&result), Vec::<&str>::new(), "{result}");
    assert_eq!(shade_smooth_values(&scene)?, [json!(true), json!(false)]);
    Ok(())
}
