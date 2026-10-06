---
name: potter
description: Model, sculpt, rig, animate, render and exchange 3D scenes headlessly with the `pot` CLI (potter) using typed JSON operations. Use when asked to create or edit 3D models/scenes, make previews/renders, inspect or pick geometry, or convert to/from .blend, GLB/glTF, USD, FBX, OBJ, PLY, STL, BVH, Alembic, SVG or PDF.
---

# potter (`pot`)

Headless, AI-oriented Blender alternative. All edits are versioned typed JSON batches applied atomically to a project directory. No GUI. Only `.blend` import/export launches Headless Blender.

## Golden rules

- Always pass `--json`. stdout is exactly one JSON envelope: `{schema_version, command, ok, scene:{id,path,revision,hash}, result, warnings, error:{code,message,details}}`. Branch on `ok` / `error.code`, never on text.
- Every mutating call needs the current revision: read `scene.revision` from the last envelope (or `pot inspect`) and send it as `base_revision`. Mismatch → `REVISION_CONFLICT` (exit 5); re-inspect, rebuild the batch, retry. Never re-send the same batch after an unknown outcome without inspecting first.
- IDs are yours to choose: `^[a-z][a-z0-9_-]{0,63}$`. Names are display only. ID clash → `ID_EXISTS`; no auto-rename.
- Unknown fields/flags/enums are errors. Ask the tool, don't guess: `pot schema --op <op> --json` returns the exact JSON Schema of one operation; `pot schema --kind capabilities --json` lists what is supported / `not_supported` (with reason).
- Units: meters, Z-up, right-handed (+X right, +Y back, +Z up). Rotations: `rotation_deg:[x,y,z]` (applied Rz·Ry·Rx) or `rotation_quat:[x,y,z,w]` — never both.
- A batch is all-or-nothing. On failure `error.details.operation_index` and `error.details.pointer` (JSON Pointer into your batch) say what to fix.

## Loop

```sh
pot init ./scene --json                                   # revision 0
pot apply ./scene --file ops.json --preview iso --json     # commit + render preview
pot inspect ./scene --json                                # revision, objects, bounds, dimensions
pot preview ./scene --views front,right,top,iso --size 768 --out ./previews --json
pot pick ./scene --render ./previews/iso.manifest.json --pixel 320,240 --json
pot validate ./scene --json
pot export ./scene --format glb --out model.glb --json
```

`apply --file -` reads the batch from stdin. `--dry-run` validates and returns the diff without saving (not combinable with `--preview`). Look at preview PNGs (paths are in `result.previews[*]`) to check your work visually, then iterate.

## Operation batch

```json
{
  "schema_version": 1,
  "base_revision": 0,
  "operations": [
    {"op": "material.create", "id": "clay", "name": "Clay",
     "base_color": [0.65, 0.32, 0.18, 1], "metallic": 0, "roughness": 0.8},
    {"op": "node.create", "id": "assembly", "kind": "group"},
    {"op": "node.create", "id": "body", "name": "Body", "kind": "box",
     "parent": "assembly", "tags": ["body"],
     "params": {"size": 1},
     "transform": {"translation": [0, 0, 0.4], "scale": [1, 0.6, 0.8]},
     "material": "clay"}
  ]
}
```

Update (absolute values; omitted fields kept):

```json
{"op": "node.update", "target": {"id": "body"}, "scope": "single_user",
 "set": {"params": {"size": 1.2}, "transform": {"rotation_deg": [0, 0, 15]}}}
```

Targets: `{"id":"x"}` or `{"tag":"t"}` (must match exactly one, else `TARGET_NOT_FOUND` / `AMBIGUOUS_TARGET`), `{"tag":"leg","many":true}` where the op allows it. Element edits add `"elements":{"domain":"vertex|edge|face","ids":["v1","e3","f5"]}` (IDs from `inspect` or `pick`).

Shared geometry: editing a Data-Block used by several objects requires `"scope":"shared"` (all users) or `"scope":"single_user"` (detach this object); omitting it → `SHARED_DATA_REQUIRES_SCOPE`. `node.duplicate` with `"mode":"linked"` shares geometry.

Primitive kinds for `node.create`: `group` (empty), `box`, `uv_sphere`/`sphere`, `cylinder`, `plane`, `cone`, `torus`, `icosphere`, `circle`, `grid`; also `camera`, `light`, `curve`, `surface`, `text`, `metaball`, `lattice`, `pointcloud`, `volume`, `armature`, `grease_pencil`, `collection_instance`. Get params per kind from `pot schema --op node.create --json`.

## Op families (see `pot schema --json` for all)

| Area | Ops |
| --- | --- |
| objects/scene | `node.create/update/delete/duplicate/parent`, `data.make_single_user`, `collection.*`, `scene.create/update/marker_add/marker_remove` |
| mesh edit | `mesh.extrude/inset/bevel/loop_cut/bridge/weld/split/dissolve/delete/fill/subdivide/triangulate/knife/bisect/remesh/mirror/symmetrize/flip_normals/transform_elements/set_attribute` |
| modifiers / nodes | `modifier.create/update/reorder/delete/apply/apply_as_shape_key/bind/unbind` (mirror, array, subdivision, solidify, bevel, decimate, boolean, remesh, skin, wireframe, surface_deform, mesh_deform, laplacian_deform, ocean, cloth, particle_instance, nodes, …; params use Blender RNA names), `graph.*` (Geometry Nodes / shader / compositor graphs) |
| sculpt / paint / UV | `sculpt.stroke/mask/face_set/remesh_voxel/multires`, `paint.vertex/weight/texture`, `uv.unwrap/pin/pack/transform`, `image.*` |
| material | `material.create/update/delete` (simple PBR fields or a full shader graph) |
| rig / animation | `bone.*`, `pose.set/reset`, `constraint.*`, `driver.*`, `shape_key.*`, `vertex_group.*`, `action.*`, `keyframe.insert/delete`, `fcurve.update`, `nla.*` |
| other data | `curve.*`, `surface.create`, `text_object.*`, `metaball.*`, `lattice.*`, `pointcloud.*`, `volume.*`, `grease_pencil.*` |
| look / output | `camera.*`, `light.*`, `world.*`, `render.update/passes_update`, `color.update`, `compositor.*`, `sequencer.*`, `mask.*`, `tracking.*` |
| physics | `physics.world.update`, `physics.rigid_body.*`, `physics.force_field.*`, `simulation.settings.update` |
| assets | `library.link/append/override/reload/relocate`, `resource.pack/unpack`, `asset.*`, `text.*`, `property.*`, `extension.register` |

- `pot schema --op modifier.create --json` shows per-type modifier `params` fields.

Example animation + camera (keyframes need an Action assigned to the node first, else `INVALID_OPERATION`):

```json
{"op": "camera.create", "id": "cam", "transform": {"translation": [4, -4, 3], "rotation_deg": [63, 0, 45]}},
{"op": "scene.update", "target": {"id": "scene_main"}, "set": {"camera": "cam"}},
{"op": "light.create", "id": "sun", "light_type": "sun", "energy": 3},
{"op": "action.create", "id": "body_anim"},
{"op": "node.update", "target": {"id": "body"}, "set": {"action": "body_anim"}},
{"op": "keyframe.insert", "target": {"id": "body"}, "path": "transform.translation", "index": 2, "frame": 1, "value": 0.4, "interpolation": "bezier"},
{"op": "keyframe.insert", "target": {"id": "body"}, "path": "transform.translation", "index": 2, "frame": 24, "value": 2.0, "interpolation": "bezier"}
```

Verify exact field names with `pot schema --op <op> --json` before first use.

## Observe

- `pot inspect <scene> [--id X | --tag T] [--frame F] [--scene-id S] --json`: `result.items[*]` has kind, parent/children, `bounds {min,max}` (world AABB), `dimensions`, raw/evaluated transforms, data sharing, modifier stack, geometry counts. `result.summary` counts everything (hidden/unused included).
- Final world placement is `transform.world` (including constraints, parents, and drivers); `transform.evaluated` is the local animated value, like Blender's `location`.
- `pot inspect <scene> --features --json`: capability of every feature (create/edit/evaluate/render/import/export, `not_supported` with reason).
- `pot preview`: views `front,back,right,left,top,bottom,iso` (orthographic) or `--camera <id>`; `--mode solid|beauty|wire|normal|depth|id`; `--frame F`. Writes `<view>.png`, `.ids.bin`, `.depth.bin`, `.elements.bin`, `.manifest.json`. Existing different output needs `--overwrite`.
- `pot pick --render <manifest> --pixel x,y [--domain object|face|vertex|edge|stroke|point]`: pixel in full-size PNG coordinates (top-left origin). Returns `hit`, `target` (ready-to-use selector + `snapshot_hash`), `world_position`, `world_normal`. `STALE_RENDER` → re-preview after edits.
- `pot history|undo|redo <scene> --base-revision N [--steps K] --json`: undo/redo create a NEW revision restoring the old state.

## Output

- `pot render <scene> --frames 1:24[:step] --format png|exr --out ./frames --json` uses the scene camera (`TARGET_NOT_FOUND` if none) and `render.*` settings (resolution, samples, seed, engine `path|realtime`).
- `pot bake <scene> --kind animation|geometry|simulation|texture --frames 1:120 --out ./bake --json`.
- `pot export <scene> --format glb|gltf|obj|stl|ply|usda|usdc|usdz|fbx|abc|bvh|svg|pdf|blend --out <path> --json`. Losses (feature the format can't hold) are refused with `UNREPRESENTABLE_FEATURE` and listed in `result.losses`; add `--allow-lossy` only if the loss is acceptable. `--overwrite` to replace. SVG/PDF accept `--view front|…`.
- `pot import <scene> --file in.glb --format glb --base-revision N [--mode append|replace] --json`. Append remaps colliding IDs (`result.id_mappings`).

## Blender (.blend)

- `pot import ./scene --file original.blend --format blend --mode replace --base-revision 0 --json` imports all scenes, hidden and unused data editable (modifiers/nodes/rig/animation stay live, not baked).
- `pot export ./scene --format blend --out model.blend --json` rebuilds a full .blend (IDs stored as custom property `potter.id`).
- Blender executable: `--blender <path>` → `POTTER_BLENDER` → `PATH` → `/Applications/Blender.app/Contents/MacOS/Blender`. Requires Blender 5.2.x (`BLENDER_NOT_FOUND`, `BLENDER_VERSION_UNSUPPORTED`).
- Unsupported Modifier/constraint evaluation (e.g. Skin branch hulls, remesh on open meshes, segmented B-Bones) is reported per command: targeted `inspect` (`--id`/`--tag`) returns `UNSUPPORTED_FEATURE` with `error.details.feature_id`, while scene-wide `inspect` continues and attaches a typed per-item error to each affected Object. `preview` renders every evaluable Object and warns for each skipped Object with its `feature_id`. By contrast, `render`, `bake`, and non-`.blend` exports stay strict: if evaluation hits an unsupported Object, they fail with typed `UNSUPPORTED_FEATURE` including `error.details.feature_id` and `error.details.node_id`; they never publish partial output. `.blend` export still preserves these features live for Blender to evaluate after reopen.
- Bound deformers (`surface_deform`, `mesh_deform`, `laplacian_deform`) need `modifier.bind` after the target/cage is in its rest shape.
- `modifier.apply` is refused when the mesh has shape keys (same as Blender); use `modifier.apply_as_shape_key` instead.
- Potter-authored camera/object tracking solves export as tracks/markers/intrinsics only: strict export reports `blender.movieclip.reconstruction`; pass `--allow-lossy` to accept it.

## Error codes → action

| exit | codes | do |
| --- | --- | --- |
| 2 | `INVALID_ARGUMENT`, `INVALID_OPERATION`, `UNSUPPORTED_VERSION`, `LIMIT_EXCEEDED` | fix the batch/flags at `details.pointer`; check `pot schema --op` |
| 3 | `SCENE_NOT_FOUND`, `FILE_NOT_FOUND`, `TARGET_NOT_FOUND`, `BLENDER_NOT_FOUND`, `DEPENDENCY_MISSING` | inspect to find real IDs/paths |
| 4 | `SCENE_INVALID`, `EVALUATION_FAILED`, `VALIDATION_FAILED`, `UNSUPPORTED_FEATURE`, `UNREPRESENTABLE_FEATURE` | read `details.feature_id`; choose a supported alternative |
| 5 | `REVISION_CONFLICT`, `SCENE_BUSY`, `AMBIGUOUS_TARGET`, `SHARED_DATA_REQUIRES_SCOPE`, `ID_EXISTS`, `STALE_RENDER`, `OUTPUT_EXISTS` | re-inspect / add scope / new ID / `--overwrite` |
| 6 | `IO_ERROR`, `IMPORT_FAILED`, `EXPORT_FAILED`, `RENDER_FAILED`, `BAKE_FAILED` | check paths/permissions, `error.details` |

## Modeling tips

- Build from primitives + modifiers first (non-destructive), `modifier.apply` only when you need element-level edits.
- Put related parts under a `group` node and tag them; move the group, not each part.
- Check `dimensions` in `inspect` against the intended real-world size (meters) after each batch.
- Use `preview --views front,right,top,iso` to catch misplaced parts; use `pick` to get element IDs for `mesh.*` edits instead of guessing.
