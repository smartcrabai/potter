//! Blender-side modifier order and pin-to-last flags after `.blend` export and re-import.
//!
//! Each stack is built through Potter's own operations, so Potter's insertion and pin rules decide
//! its order. The exported `.blend` is reopened in Blender 5.2.x and its stack must equal Potter's
//! names and pin flags. Requires Blender 5.2.x; skipped when it is missing, and failing instead when
//! `POTTER_REQUIRE_BLENDER` is set.

use std::{
    env,
    error::Error,
    fs,
    path::{Path, PathBuf},
    process::Command,
};

#[path = "common/blender_checked.rs"]
mod blender_checked;
#[path = "common/blender_script.rs"]
mod blender_script;
#[path = "common/process.rs"]
mod process;

use blender_checked::blender_executable;
use blender_script::run_blender_script;
use potter_core::{
    error::ErrorCode,
    eval::EvaluationContext,
    exchange::{
        ExportOptions,
        blend::{export_blend, import_blend},
    },
    model::{Id, SceneDoc},
    ops,
    store::Project,
};
use serde_json::{Value, json};
use tempfile::tempdir;

type TestResult<T> = Result<T, Box<dyn Error>>;

const SCENE_ID: &str = "00000000-0000-4000-8000-000000000000";
const OBJECT: &str = "stack";

/// One Potter operation on the stack under test.
#[derive(Clone, Copy)]
enum Step {
    Create {
        name: &'static str,
        kind: &'static str,
        pinned: bool,
    },
    Pin {
        name: &'static str,
        pinned: bool,
    },
}

use Step::{Create, Pin};

struct Case {
    name: &'static str,
    steps: &'static [Step],
    /// Potter's stack after the steps, bottom to top, as (name, `use_pin_to_last`).
    expected: &'static [(&'static str, bool)],
}

/// Stacks whose order Blender reaches only through a specific construction order or pin sequence.
const CASES: &[Case] = &[
    Case {
        name: "array_then_softbody",
        steps: &[
            Create {
                name: "Softbody",
                kind: "soft_body",
                pinned: true,
            },
            Create {
                name: "Array",
                kind: "array",
                pinned: false,
            },
            Pin {
                name: "Softbody",
                pinned: false,
            },
        ],
        expected: &[("Array", false), ("Softbody", false)],
    },
    Case {
        name: "array_then_multires",
        steps: &[
            Create {
                name: "Multires",
                kind: "multires",
                pinned: true,
            },
            Create {
                name: "Array",
                kind: "array",
                pinned: false,
            },
            Pin {
                name: "Multires",
                pinned: false,
            },
        ],
        expected: &[("Array", false), ("Multires", false)],
    },
    Case {
        name: "pinned_array_then_cloth",
        steps: &[
            Create {
                name: "Array",
                kind: "array",
                pinned: true,
            },
            Create {
                name: "Cloth",
                kind: "cloth",
                pinned: false,
            },
        ],
        expected: &[("Cloth", false), ("Array", true)],
    },
    Case {
        name: "pinned_smooth_then_hook",
        steps: &[
            Create {
                name: "Smooth",
                kind: "smooth",
                pinned: true,
            },
            Create {
                name: "Hook",
                kind: "hook",
                pinned: false,
            },
        ],
        expected: &[("Smooth", true), ("Hook", false)],
    },
    Case {
        name: "hook_then_pinned_smooth",
        steps: &[
            Create {
                name: "Hook",
                kind: "hook",
                pinned: false,
            },
            Create {
                name: "Smooth",
                kind: "smooth",
                pinned: true,
            },
        ],
        expected: &[("Hook", false), ("Smooth", true)],
    },
    Case {
        name: "multiple_pins",
        steps: &[
            Create {
                name: "Bevel",
                kind: "bevel",
                pinned: true,
            },
            Create {
                name: "Array",
                kind: "array",
                pinned: true,
            },
        ],
        expected: &[("Array", true), ("Bevel", true)],
    },
];

/// Blender 5.2.2 fixture: a Softbody pinned while alone, then an Array, then the pin cleared.
/// Reopening it in Potter must keep the stack order it was saved with.
const SOURCE_FIXTURE: &str = r#"
import bpy
import os
import sys

root = os.path.realpath(sys.argv[sys.argv.index("--") + 1])
bpy.ops.wm.read_factory_settings(use_empty=True)
mesh = bpy.data.meshes.new("StackMesh")
mesh.from_pydata([(-1.0, -1.0, 0.0), (1.0, -1.0, 0.0), (1.0, 1.0, 0.0), (-1.0, 1.0, 0.0)],
                 [], [(0, 1, 2, 3)])
obj = bpy.data.objects.new("stack", mesh)
bpy.context.scene.collection.objects.link(obj)
softbody = obj.modifiers.new("Softbody", "SOFT_BODY")
softbody.use_pin_to_last = True
obj.modifiers.new("Array", "ARRAY")
bpy.ops.wm.save_as_mainfile(filepath=os.path.join(root, "source.blend"))
"#;

/// Reads the object's modifier stack from a saved `.blend` as [name, type, pinned] rows.
const READBACK: &str = r#"
import bpy
import json
import sys

argv = sys.argv[sys.argv.index("--") + 1:]
bpy.ops.wm.open_mainfile(filepath=argv[0])
obj = bpy.data.objects[argv[1]]
with open(argv[2], "w", encoding="utf-8") as output:
    json.dump([[m.name, m.type, bool(m.use_pin_to_last)] for m in obj.modifiers], output)
"#;

/// Replaces a wrong-type target modifier, removes an extra modifier, and preserves a pinned target.
const EXISTING_STACK_RESTORE: &str = r#"
import bpy
import json
import os
import sys

root = os.path.realpath(sys.argv[sys.argv.index("--") + 1])
source = open(os.path.join(root, "blend_bridge_source.py"), encoding="utf-8").read()
start = source.index("def _create_modifier(")
end = source.index("def _ensure_dynamic_paint_role(")
namespace = {"bpy": bpy, "blender_modifier_type": lambda value: value}
exec(source[start:end], namespace)

mesh = bpy.data.meshes.new("ExistingStackMesh")
obj = bpy.data.objects.new("existing_stack", mesh)
obj.modifiers.new("Extra", "BEVEL")
obj.modifiers.new("Hook", "ARRAY")
smooth = obj.modifiers.new("Smooth", "SMOOTH")
smooth.use_pin_to_last = True
namespace["_restore_modifier_stack"](obj, [
    {"name": "Hook", "type": "HOOK", "params": {}},
    {"name": "Smooth", "type": "SMOOTH", "params": {"use_pin_to_last": True}},
])
rows = [[modifier.name, modifier.type, bool(modifier.use_pin_to_last)] for modifier in obj.modifiers]
with open(os.path.join(root, "existing_stack.json"), "w", encoding="utf-8") as output:
    json.dump(rows, output)
"#;

fn require_blender() -> bool {
    env::var_os("POTTER_REQUIRE_BLENDER").is_some_and(|value| !value.is_empty())
}

/// Returns Blender 5.2.x, or `None` (skip) when it is missing. With `POTTER_REQUIRE_BLENDER` set,
/// a missing or wrong-version Blender fails the test.
fn blender_5_2() -> TestResult<Option<PathBuf>> {
    let Some(blender) = blender_executable() else {
        assert!(
            !require_blender(),
            "POTTER_REQUIRE_BLENDER is set but Blender is not available"
        );
        eprintln!("skipping modifier order checks: Blender is unavailable");
        return Ok(None);
    };
    let mut command = Command::new(&blender);
    command.arg("--version");
    let output = process::run_guarded(command)?;
    let version = String::from_utf8_lossy(&output.stdout)
        .lines()
        .next()
        .unwrap_or_default()
        .to_owned();
    if version.starts_with("Blender 5.2") {
        return Ok(Some(blender));
    }
    assert!(
        !require_blender(),
        "modifier order checks need Blender 5.2.x, found `{version}`"
    );
    eprintln!("skipping modifier order checks: found `{version}`, need Blender 5.2.x");
    Ok(None)
}

fn apply(doc: &SceneDoc, operation: &Value) -> TestResult<SceneDoc> {
    let batch = json!({
        "schema_version": 1,
        "base_revision": doc.revision,
        "operations": [operation],
    });
    Ok(ops::apply_batch(doc, &batch)?.doc)
}

fn stack_doc(steps: &[Step]) -> TestResult<SceneDoc> {
    let mut doc = apply(
        &SceneDoc::new(SCENE_ID.to_owned()),
        &json!({"op": "node.create", "id": OBJECT, "kind": "box", "params": {"size": 2.0}}),
    )?;
    for step in steps {
        let operation = match *step {
            Create { name, kind, pinned } => json!({
                "op": "modifier.create",
                "target": {"id": OBJECT},
                "id": name.to_ascii_lowercase(),
                "name": name,
                "type": kind,
                "params": {"use_pin_to_last": pinned},
            }),
            Pin { name, pinned } => json!({
                "op": "modifier.update",
                "target": {"id": OBJECT},
                "id": name.to_ascii_lowercase(),
                "set": {"params": {"use_pin_to_last": pinned}},
            }),
        };
        doc = apply(&doc, &operation)?;
    }
    Ok(doc)
}

/// Potter's stack for `OBJECT`, bottom to top, as (name, `use_pin_to_last`).
fn potter_stack(doc: &SceneDoc) -> TestResult<Vec<(String, bool)>> {
    potter_stack_for(doc, OBJECT)
}

fn potter_stack_for(doc: &SceneDoc, object_id: &str) -> TestResult<Vec<(String, bool)>> {
    let node = doc
        .nodes
        .get(&Id::new(object_id.to_owned())?)
        .ok_or("stack node is missing")?;
    Ok(node
        .modifiers
        .iter()
        .map(|modifier| {
            let pinned = modifier
                .params
                .get("use_pin_to_last")
                .and_then(Value::as_bool)
                .unwrap_or(false);
            (modifier.name.clone(), pinned)
        })
        .collect())
}

fn blender_stack(blender: &Path, root: &Path, blend: &Path) -> TestResult<Vec<(String, bool)>> {
    let script = root.join("readback.py");
    fs::write(&script, READBACK)?;
    let output_path = root.join("readback.json");
    let mut command = Command::new(blender);
    command
        .args(["--background", "--factory-startup", "--python"])
        .arg(&script)
        .arg("--")
        .arg(blend)
        .arg(OBJECT)
        .arg(&output_path);
    let output = process::run_guarded(command)?;
    assert!(
        output.status.success(),
        "Blender readback failed: stdout={} stderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let rows: Value = serde_json::from_slice(&fs::read(output_path)?)?;
    rows.as_array()
        .ok_or("Blender stack must be an array")?
        .iter()
        .map(|row| {
            let name = row
                .get(0)
                .and_then(Value::as_str)
                .ok_or("Blender modifier name is missing")?;
            let pinned = row
                .get(2)
                .and_then(Value::as_bool)
                .ok_or("Blender pin flag is missing")?;
            Ok((name.to_owned(), pinned))
        })
        .collect()
}

#[test]
fn blender_stack_order_and_pins_match_potter_for_each_stack() -> TestResult<()> {
    let Some(blender) = blender_5_2()? else {
        return Ok(());
    };
    let directory = tempdir()?;
    let root = directory.path();
    let project = Project::init(root.join("project"))?;
    let context = EvaluationContext::default();
    for case in CASES {
        let doc = stack_doc(case.steps)?;
        let potter = potter_stack(&doc)?;
        let expected: Vec<(String, bool)> = case
            .expected
            .iter()
            .map(|(name, pinned)| ((*name).to_owned(), *pinned))
            .collect();
        assert_eq!(potter, expected, "{}: Potter stack", case.name);

        let blend = root.join(format!("{}.blend", case.name));
        export_blend(
            &doc,
            &project,
            &blend,
            &ExportOptions {
                allow_lossy: false,
                pack: false,
                blender: Some(blender.as_path()),
                context: &context,
            },
        )?;
        assert_eq!(
            blender_stack(&blender, root, &blend)?,
            potter,
            "{}: Blender stack order and pins",
            case.name
        );
    }
    Ok(())
}

#[test]
fn potter_order_blender_cannot_represent_fails_with_precise_error() -> TestResult<()> {
    let Some(blender) = blender_5_2()? else {
        return Ok(());
    };
    let directory = tempdir()?;
    let project = Project::init(directory.path().join("project"))?;
    let context = EvaluationContext::default();
    let mut doc = stack_doc(&[
        Create {
            name: "Array",
            kind: "array",
            pinned: true,
        },
        Create {
            name: "Cloth",
            kind: "cloth",
            pinned: false,
        },
    ])?;
    // Potter's insertion keeps the unpinned cloth in front of the pinned array. Swapping them
    // gives a pinned modifier with an unpinned one after it, which no Blender sequence reaches.
    doc.nodes
        .get_mut(&Id::new(OBJECT)?)
        .ok_or("stack node is missing")?
        .modifiers
        .swap(0, 1);
    let blend = directory.path().join("unrepresentable.blend");
    let Err(error) = export_blend(
        &doc,
        &project,
        &blend,
        &ExportOptions {
            allow_lossy: false,
            pack: false,
            blender: Some(blender.as_path()),
            context: &context,
        },
    ) else {
        return Err("export must fail for an order Blender cannot represent".into());
    };
    assert_eq!(error.code, ErrorCode::ExportFailed);
    let stderr = error.details["stderr"].as_str().unwrap_or_default();
    assert!(
        stderr.contains(
            "cannot reproduce modifier order on stack: the order is not reachable with Blender operations; first target position 1 (Cloth, pinned to last=False) is never reached"
        ),
        "unexpected export error: {stderr}"
    );
    assert!(!blend.exists(), "a failed export must not write the .blend");
    Ok(())
}

#[test]
fn imported_blender_stack_order_survives_reexport() -> TestResult<()> {
    let Some(blender) = blender_5_2()? else {
        return Ok(());
    };
    let directory = tempdir()?;
    let root = directory.path();
    run_blender_script(
        &blender,
        "source_fixture.py",
        SOURCE_FIXTURE,
        root,
        "Blender source fixture failed",
    )?;
    let source = root.join("source.blend");
    let imported = import_blend(&source, Some(blender.as_path()))?;
    let imported_id = imported
        .id_mappings
        .get("Object:stack")
        .and_then(Value::as_str)
        .ok_or("Blender stack object ID mapping is missing")?;
    let imported_stack = potter_stack_for(&imported.doc, imported_id)?;
    assert_eq!(
        imported_stack,
        vec![("Array".to_owned(), false), ("Softbody".to_owned(), true)],
        "imported Potter stack"
    );

    let project = Project::init(root.join("project"))?;
    let context = EvaluationContext::default();
    let reexport = root.join("reexport.blend");
    // Imported scenes can carry unrelated compatibility losses; the order check is what matters.
    export_blend(
        &imported.doc,
        &project,
        &reexport,
        &ExportOptions {
            allow_lossy: true,
            pack: false,
            blender: Some(blender.as_path()),
            context: &context,
        },
    )?;
    assert_eq!(
        blender_stack(&blender, root, &reexport)?,
        imported_stack,
        "re-exported Blender stack order and pins"
    );
    Ok(())
}

#[test]
fn blender_existing_stack_replaces_wrong_type_and_removes_stale_modifier_before_planning()
-> TestResult<()> {
    let Some(blender) = blender_5_2()? else {
        return Ok(());
    };
    let directory = tempdir()?;
    let root = directory.path();
    let bridge_source = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("crates/potter-core/src/exchange/blend_bridge.py");
    fs::write(
        root.join("blend_bridge_source.py"),
        fs::read(bridge_source)?,
    )?;
    run_blender_script(
        &blender,
        "existing_stack.py",
        EXISTING_STACK_RESTORE,
        root,
        "Blender existing-stack restore failed",
    )?;
    let result: Value = serde_json::from_slice(&fs::read(root.join("existing_stack.json"))?)?;
    assert_eq!(
        result,
        json!([["Hook", "HOOK", false], ["Smooth", "SMOOTH", true]])
    );
    Ok(())
}
