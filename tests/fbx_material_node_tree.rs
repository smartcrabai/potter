//! FBX export of a material with a custom node tree. FBX cannot hold the graph, so strict export
//! refuses it as a `format.fbx.material_node_tree` loss, and `--allow-lossy` writes the material's
//! mirror fields (`DiffuseColor` and the other FBX material properties) instead.

#[path = "common/native_oracle.rs"]
#[expect(
    dead_code,
    reason = "this integration binary uses a subset of the shared native oracle helpers"
)]
mod native_oracle;

use std::{fs, path::PathBuf};

use native_oracle::{TestResult, apply_operations, loss_features, pot, pot_json};
use serde_json::{Value, json};

/// A project with one box whose material has a shader graph beyond the default one: an unlinked
/// Math node the FBX writer cannot represent.
fn custom_graph_project(root: &std::path::Path) -> TestResult<PathBuf> {
    let project = root.join("project");
    pot_json(|command| {
        command.arg("init").arg(&project);
    })?;
    apply_operations(
        &project,
        root,
        json!([
            {
                "op": "material.create",
                "id": "custom",
                "name": "Custom",
                "graph": {
                    "kind": "shader",
                    "nodes": {
                        "out": {"type": "OutputMaterial"},
                        "principled": {
                            "type": "ShaderNodeBsdfPrincipled",
                            "inputs": {"Base Color": [0.2, 0.4, 0.6, 1.0]}
                        },
                        "spare": {"type": "ShaderNodeMath"}
                    },
                    "links": [{
                        "from_node": "principled", "from_socket": "BSDF",
                        "to_node": "out", "to_socket": "Surface"
                    }]
                }
            },
            {
                "op": "node.create",
                "id": "cube",
                "kind": "box",
                "params": {"size": 1.0},
                "material": "custom"
            }
        ]),
    )?;
    Ok(project)
}

#[test]
fn strict_fbx_export_refuses_a_material_node_tree() -> TestResult {
    let directory = tempfile::tempdir()?;
    let project = custom_graph_project(directory.path())?;
    let out = directory.path().join("strict.fbx");
    let output = pot()
        .arg("export")
        .arg(&project)
        .args(["--format", "fbx", "--out"])
        .arg(&out)
        .arg("--json")
        .output()?;
    assert!(
        !output.status.success(),
        "strict FBX export accepted a material node tree"
    );
    let report: Value = serde_json::from_slice(&output.stdout)?;
    assert!(
        loss_features(&report).contains(&"format.fbx.material_node_tree".to_owned()),
        "strict FBX export did not name the node-tree loss: {report}"
    );
    assert!(!out.exists(), "strict FBX export wrote a file");
    Ok(())
}

#[test]
fn allow_lossy_fbx_export_writes_the_mirror_fields() -> TestResult {
    let directory = tempfile::tempdir()?;
    let project = custom_graph_project(directory.path())?;
    let out = directory.path().join("allow.fbx");
    let output = pot()
        .arg("export")
        .arg(&project)
        .args(["--format", "fbx", "--out"])
        .arg(&out)
        .arg("--allow-lossy")
        .arg("--json")
        .output()?;
    assert!(
        output.status.success(),
        "--allow-lossy FBX export failed: {}",
        String::from_utf8_lossy(&output.stdout)
    );
    let report: Value = serde_json::from_slice(&output.stdout)?;
    assert!(
        loss_features(&report).contains(&"format.fbx.material_node_tree".to_owned()),
        "--allow-lossy export did not report the node-tree loss: {report}"
    );
    let text = fs::read_to_string(&out)?;
    assert!(text.contains("Material: "), "FBX has no material object");
    assert!(
        text.contains("\"DiffuseColor\""),
        "FBX material lacks its DiffuseColor mirror field"
    );
    Ok(())
}
