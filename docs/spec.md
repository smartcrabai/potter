# potter Specification

Status: Final specification. Before implementation. Spec version: 2. Updated: 2026-10-05.

potter is a person who makes objects from clay. It is a headless, AI-friendly Blender alternative. All features are in scope for implementation and validation in the first release. There are no phased releases by feature.

Completion criteria: Modeling, sculpting, paint, Geometry Nodes, rigging, animation, physics, rendering, compositing, video editing, asset management, and Blender I/O MUST be completed using only the CLI and typed JSON. Every item in the [Blender Compatibility requirements](blender-compatibility.md) is REQUIRED.

## 1. Decisions

| Item | Decision |
| --- | --- |
| Product name / executable name | `potter` / `pot` |
| Core | Rust. Use the existing Cargo project |
| Creation and evaluation engine | potter's own engine, implemented independently. Blender is not required for normal editing, evaluation, simulation, or rendering |
| Blender usage | Headless Blend Adapter for `.blend` reading/writing only. It MUST NOT be used as an alternative backend for normal processing |
| Runtime environment | No GUI or display required. All basic workflows run on CPU. GPU is optional acceleration |
| Editing input | Versioned typed JSON. Natural-language interpretation is the responsibility of the calling AI |
| Source of truth | Creation structure in `scene.json` plus content-hash references to assets, binaries, and compatibility payloads |
| Evaluation result | Snapshot supporting mesh, curve, hair, point, volume, instance, pose, simulation, and render pass data |
| Blender Compatibility baseline version | Blender 5.2 LTS. Initial validation build: 5.2.2. Fix types, properties, and versions in the profile |
| I/O | `.blend`, GLB / glTF, USD / USDZ, Alembic, FBX, OBJ, PLY, STL, BVH, SVG, PDF |
| Observability | Retrieve structure, dimensions, dependencies, sharing, features, frame evaluation, images, and pixel picks as JSON |
| Target OS | macOS arm64, Linux x86_64 / arm64, Windows x86_64 / arm64 |

Feature scope: mesh editing and all standard Modifiers; curve / surface / text / metaball / lattice / point / volume; Geometry Nodes; sculpting and texture / vertex / weight paint; UV / UDIM / shader; armature / constraint / driver / shape key / Action / NLA; physics; Grease Pencil; camera / light / world; PBR / path tracing / real-time rendering / color management; compositor; tracking / mask; Video Sequencer; asset / library / override.

See the [Blender Compatibility requirements](blender-compatibility.md) as the source of truth for specific gap remediation, round-trip conditions, and exhaustive machine checks. Standard features that are "save-only" or "not implemented" do not meet the first-release requirement. Replace GUI actions with CLI inputs for numbers, targets, strokes, and so on.

## 2. Basic workflow

The following is an example of use after implementation. The first `operations.json` is the example in Section 5.

```sh
pot init ./scene

pot apply ./scene --file operations.json --preview iso --json
pot inspect ./scene --tag body --json

pot preview ./scene \
  --views front,right,top,iso \
  --size 768 --out ./previews --json

pot pick ./scene \
  --render ./previews/iso.manifest.json \
  --pixel 320,240 --json

pot validate ./scene --json
pot export ./scene --format glb --out model.glb --json
pot export ./scene --format blend --out model.blend --json

# Read all Blender Scenes and production data
pot init ./from-blender
pot import ./from-blender \
  --file original.blend --format blend \
  --mode replace --base-revision 0 --json

# Evaluate and capture a specific frame and Scene
pot inspect ./from-blender --scene-id scene_main --frame 24 --json
pot preview ./from-blender \
  --scene-id scene_main --frame 24 --mode beauty --views iso --json

# Render animation. Return paths to generated image sequences / video
pot render ./scene --frames 1:120 --format exr --out ./frames --json

# Bake simulation, texture, geometry, and animation
pot bake ./scene --kind simulation --frames 1:120 --out ./bakes --json

pot assets ./scene --check --json
pot schema --kind operations --json
pot schema --kind capabilities --json
```

AI iteration: retrieve revision, targets, and sharing relationships with `inspect` -> create operations JSON -> `apply --preview` -> image / `pick` -> edit again. After completion, run `validate` and `export`. Additional features use the same editing and observability contract.

## 3. CLI contract

```text
pot init <scene> [--json]
pot history <scene> [--json]
pot undo <scene> --base-revision <n> [--steps <n>] [--json]
pot redo <scene> --base-revision <n> [--steps <n>] [--json]
pot apply <scene> --file <path|-> [--preview <views>] [--size <px>] [--dry-run] [--json]
pot import <scene> --file <path> --format <format> --base-revision <n>
  [--mode append|replace] [--asset-policy copy|link] [--allow-lossy] [--dry-run] [--json]
pot inspect <scene> [--id <id> | --tag <tag>] [--features] [context] [--json]
pot preview <scene> [--views <views> | --camera <id>] [--mode <mode>]
  [--size <px>] [--out <dir>] [--overwrite] [context] [--json]
pot pick <scene> --render <manifest> --pixel <x,y>
  [--domain object|vertex|edge|face|bone|stroke|point] [--json]
pot validate <scene> [--strict] [--format <format>] [context] [--json]
pot export <scene> --format <format> --out <path>
  [--allow-lossy] [--pack] [--overwrite] [context] [--json]
pot render <scene> [--camera <id>] [--engine path|realtime] [--device cpu|gpu]
  [--frames <start:end[:step]>] --format <format> --out <dir>
  [--overwrite] [context] [--json]
pot bake <scene> --kind simulation|texture|geometry|animation
  [--target <id>] [--frames <start:end[:step]>] --out <dir>
  [--overwrite] [context] [--json]
pot assets <scene> [--check] [--json]
pot schema [--kind scene|operations|preview|response|capabilities|formats]
  [--op <op>] [--json]
pot --help
pot --version

context = [--scene-id <id>] [--view-layer <id>] [--frame <number>]
.blend import/export = the above, optionally with [--blender <executable>]
```

- `<scene>` is the working directory. Select an internal Scene with `--scene-id`. If omitted, use `active_scene`. Relative argument paths are based on cwd.
- frame is a finite real number: an integer part plus a subframe. If `--frame` is omitted, use the selected Scene's current frame. frames includes both endpoints, with a default step of 1. fps / fps_base are Scene settings.
- Specify apply's evaluation context, camera, and mode in the optional `evaluation` field of the operations JSON. If omitted, use the active Scene / View Layer / current frame / solid.
- `--file -` reads JSON from stdin for apply only. Binary input for import requires a file.
- For both success and failure, `--json` writes one JSON object followed by a newline to stdout. All logs and child-process output go to stderr. No progress JSON is interleaved.
- Non-interactive. Unknown commands, flags, input fields, and enums are errors. Input fields are never silently ignored.
- `--help` and `--version` print text and exit with code 0. The same applies when combined with `--json`.
- `--views` and `--preview` use CSV. Duplicates, empty elements, and unknown views are errors.
- Preview size is an integer from 64 to 16384, defaulting to 768. Large images are processed in tiles. Render resolution, samples, passes, and codec use Scene render settings.
- The default `schema` kind is operations. Include JSON Schema Draft 2020-12 and resolve all `$ref` values locally. Specify an op to retrieve its individual operation schema.
- formats returns directions, supported features, required runtimes, and loss conditions. capabilities returns the custom engine's actual support.
- Include Snapshot selection, driver execution policy, asset resolution, and solver / renderer versions in the evaluation context. Automatic script execution from input .blend files is disabled by default.

### Exit codes

| Value | Meaning | Common error.code values |
| --- | --- | --- |
| 0 | Success. Background pick is also successful | — |
| 1 | Internal error | `INTERNAL_ERROR` |
| 2 | CLI, operation schema, version, or limit violation | `INVALID_ARGUMENT`, `INVALID_OPERATION`, `UNSUPPORTED_VERSION`, `BLENDER_VERSION_UNSUPPORTED`, `LIMIT_EXCEEDED` |
| 3 | Missing input, target, or dependency runtime | `SCENE_NOT_FOUND`, `FILE_NOT_FOUND`, `TARGET_NOT_FOUND`, `BLENDER_NOT_FOUND`, `DEPENDENCY_MISSING` |
| 4 | Invalid production data, evaluation, or compatibility | `SCENE_INVALID`, `EVALUATION_FAILED`, `RENDER_INVALID`, `VALIDATION_FAILED`, `UNSUPPORTED_FEATURE`, `UNREPRESENTABLE_FEATURE` |
| 5 | Revision, target, or output conflict | `REVISION_CONFLICT`, `SCENE_BUSY`, `AMBIGUOUS_TARGET`, `SHARED_DATA_REQUIRES_SCOPE`, `ID_EXISTS`, `STALE_RENDER`, `OUTPUT_EXISTS` |
| 6 | I/O, conversion, or generation failure | `IO_ERROR`, `IMPORT_FAILED`, `EXPORT_FAILED`, `RENDER_FAILED`, `BAKE_FAILED` |

An invalid source-of-truth schema is SCENE_INVALID. An invalid operations JSON schema is INVALID_OPERATION. An unsupported schema_version is UNSUPPORTED_VERSION. Allowing loss applies only to losses from format conversion; it does not turn missing runtimes, corruption, or unimplemented features into success.

## 4. Creation Model and Storage

### Coordinates, Transform, and Units

- The API defaults to a right-handed coordinate system: +X right, +Y back, +Z up. Length is in m, time in s, and rotation inputs MUST specify degrees, rad, or quaternion explicitly.
- Preserve Blender's unit system, scale_length, and length / mass / time settings per Scene. Convert units in the evaluation context for a Data-Block shared by Scenes with different units; do not silently break sharing. Preserve the original numeric values and unit mapping in adapter information as well.
- Numeric computation uses f64 by default. Specify storage precision for assets / volumes / images / solvers by type and format. JSON values MUST be finite, and duplicate keys are not allowed.
- Preserve local / world transforms, parent inverse, delta transform, origin, all rotation modes, negative / zero scale, and shear.
- Basic TRS is `M_local = T × R × S`. Obtain the world matrix, including parenting, Constraints, and drivers, from the evaluation graph.
- Quaternions are `[x,y,z,w]`. The default order for operation input `rotation_deg:[x,y,z]` is `Rz × Ry × Rx`. Rotation forms MUST NOT be specified simultaneously.
- Default quaternion normalization is unit normalization with positive w; if w=0, make the first nonzero value among x / y / z positive. Do not lose the original rotation mode / Euler values.
- Correct winding and normal orientation for negative scale. Singular matrices MAY be stored. An operation requiring an inverse transform returns INVALID_OPERATION with SINGULAR_TRANSFORM details.
- Matrix JSON uses 16 column-major elements. Dimensions are the world AABB `max - min` of the specified Snapshot.

### Source of Truth and Derived Data

```text
scene/
  scene.json                  # Creation structure and all source-of-truth references
  assets/sha256/<hex>/...      # Images, fonts, sounds, clips, libraries, caches, etc.
  data/sha256/<hex>/...        # Large meshes, attributes, animation, volumes, etc.
  compat/sha256/<hex>/...      # Source .blend, unknown payloads, adapter mappings
  history/sha256/<hex>/...     # Immutable revisions, operation records, undo / redo data
  .potter/
    lock
    previews/r<revision>/<render_key_hex>/...
    cache/<evaluation_key_hex>/...
    tmp/...
```

- Restoring requires scene.json and its referenced assets / data / compat / history. Only `.potter/` MAY be deleted and regenerated. Do not delete the source .blend, unknown payloads, or editing history as cache.
- Store source-of-truth blobs immutably by content hash. Register dependency hashes in scene.json. Atomically replacing scene.json commits a revision.
- Assets and binaries are copied into the project by default. Register explicitly linked external dependencies in the source of truth with URI, source path, expected hash, and status. Do not assume restoration is possible without dependencies.
- `scene_hash = "sha256:" + hex(SHA-256(JCS(scene.json)))`. Include all referenced blob hashes, the profile, and adapter mappings in the input. [RFC 8785](https://www.rfc-editor.org/rfc/rfc8785.html)
- Do not include timestamps, runtime-resolved paths, or logs in the hash if unrelated to content and evaluation. Preserve paths retained by the source file as creation data. If an external asset's content changes, stop evaluation with ASSET_CHANGED and register the new hash through import / apply.
- Assign persistent IDs to nodes, Data-Blocks, Scenes, Collections, bones, Actions, graphs, resources, etc. Names are not IDs. IDs MUST match `^[a-z][a-z0-9_-]{0,63}$`; tags use the same format.
- Names are Unicode. Validate Blender's type-specific naming constraints in the profile. Node IDs, data IDs, etc. use type-specific namespaces.
- Assign persistent element IDs to mesh vertices / edges / faces / corners, curve points, strokes, etc. Evaluation indices are valid only within a Snapshot. Return a mapping of created / split / joined / deleted elements in the operation result.
- Store all registries in ID order and tags in lexicographic order. Preserve unused, fake-user, and hidden data in the source of truth. Do not include the hash itself in the source of truth.
- Revisions range from 0 to `2^53 - 1`. scene_id is a UUID v4 fixed for the lifetime of the working directory. The JSON format uses schema_version=1. Since the specification is unpublished, migration of existing formats is not required.

### Registries and References

| Registry | Contents |
| --- | --- |
| scenes | Root collection, View Layer, unit, frame / fps, world / camera, render / compositor / sequencer settings |
| collections | Directed acyclic parent-child graph, member Objects, instance and asset information |
| nodes | Object, Data-Block references, parent, transform, visibility, material slots, Modifiers, Constraints, drivers, instances |
| data_blocks | Mesh topology / attributes, curve, surface, metaball, text, lattice, point, volume, armature, Grease Pencil, camera, light, etc. |
| materials / node_groups | All shader / Geometry / Compositor graphs, interface, sockets, links, defaults, attributes |
| actions | Action / slot / F-Curve / keyframe / NLA / driver / shape key associations |
| resources / libraries | Assets, caches, movies / sounds / fonts, links, overrides, packs, external dependencies |
| compatibility | Blender profile, source data references, ID mappings, original units / source metadata, unknown payloads, migration reports |
| history | Commit Snapshots, operation records, state mappings, undo / redo references. Preserve assets from past Snapshots as well |

Empty registries are also explicit. Standalone graphs / physics / tracking, etc. reference typed Data-Blocks from the relevant registry. Large arrays are not required to be in scene.json; use the same type inline or by hash reference.

### Geometry, Sharing, and Visibility

- Provide operations to create primitives such as boxes, spheres, cylinders, planes, cones, tori, icospheres, circles, and grids. Preserve editable descriptors and base geometry.
- The mesh source of truth consists of vertices, edges, polygons, and corner attributes. Preserve triangles / quads / n-gons and loose vertices / edges. Triangulate when observing, rendering, or converting to a format that requires it.
- UVs, normals, colors, weights, creases, seams, sharpness, and material indices are attributes on their respective domains. Vertex splitting for rendering MUST NOT change source-of-truth connectivity.
- Separate Objects from geometry. Preserve shared Data-Blocks, Collection instances, and Geometry Nodes-generated instances.
- Parents may be Objects / bones / vertices, etc. Check reference cycles separately from parenting and evaluation dependencies. Do not confuse Collection membership with parenting.
- Preserve viewport visibility, render visibility, View Layer exclusion, holdout, indirect only, and selectability separately. Do not unconditionally propagate a parent Object's hidden state to descendants.
- Changing kind requires an explicit convert operation. Report losses to the source descriptor, attributes, rig, animation, etc. No implicit convert / apply / bake.
- Materials are slot arrays plus face assignments. Support simplified base_color / metallic / roughness input through shader graphs as well. Preserve linear RGBA, alpha, all shader values, and the original graph.

init creates an empty Scene `scene_main`, Collection `collection_root`, View Layer `view_main`, current frame=1, range 1-250, fps=24, and fps_base=1. The default material is linear RGBA=[0.6,0.6,0.6,1], metallic=0, roughness=0.8, double_sided=false. camera / world are null; an empty scene is valid.

## 5. Operations JSON

Top-level required fields: schema_version, base_revision, and operations. Optional evaluation fields are scene_id, view_layer, frame, camera, mode, and execution policy. Element edits that require a Snapshot MUST also specify the evaluation hash.

First editing example. The primitive kind and params are convenience inputs; in the source of truth they are separated into an Object and shareable geometry Data-Block.

```json
{
  "schema_version": 1,
  "base_revision": 0,
  "operations": [
    {
      "op": "material.create",
      "id": "clay",
      "name": "Clay",
      "base_color": [0.65, 0.32, 0.18, 1],
      "metallic": 0,
      "roughness": 0.8
    },
    {
      "op": "node.create",
      "id": "assembly",
      "kind": "group"
    },
    {
      "op": "node.create",
      "id": "body",
      "name": "Body",
      "kind": "box",
      "parent": "assembly",
      "tags": ["body"],
      "params": {"size": [1, 0.6, 0.8]},
      "transform": {"translation": [0, 0, 0.4]},
      "material": "clay"
    }
  ]
}
```

Next editing example. Values are absolute. Use `scope` to specify the impact on sharers when editing primitive params.

```json
{
  "schema_version": 1,
  "base_revision": 1,
  "operations": [
    {
      "op": "node.update",
      "target": {"id": "body"},
      "scope": "single_user",
      "set": {
        "params": {"size": [1.2, 0.6, 0.8]},
        "transform": {"rotation_deg": [0, 0, 15]}
      }
    }
  ]
}
```

### Target, Scope, and Change Granularity

- `{"id":"body"}`: one target. `{"tag":"body"}`: requires exactly one match; zero matches yields TARGET_NOT_FOUND, multiple matches yield AMBIGUOUS_TARGET.
- `{"tag":"leg","many":true}`: all matches for an operation that supports it. Zero matches is an error. Whether many is allowed is specified in each schema.
- Combining id and tag, combining id and many, names, partial matches, and implicit selection are not allowed. Operations on a typed registry determine the target type.
- Add `elements:{"domain":"face","ids":["face_001"]}`, etc. Bones, strokes, and points are also supported. Selection by position / radius / region / attribute is also schema-defined as an explicit selector.
- A `snapshot_hash` is required when specifying an evaluation-derived index, screen coordinate, or generated element. An old Snapshot is a conflict. Do not mix persistent element IDs with evaluation indices.
- Resolve targets when an operation starts; order multiple targets by ID. Refer to preceding changes in the same batch. Forward references are not allowed.
- Edit shared Data-Blocks with `scope:"shared"` to change all users, or `scope:"single_user"` to separate the target's reference. If data is shared and scope is omitted, return SHARED_DATA_REQUIRES_SCOPE. Changes to Object-specific transforms affect only the target Object.
- Updates preserve omitted fields. Distinguish full replacement of arrays / sets from element-edit operations. Return the sharing scope, reevaluation targets, attribute interpolation, and element ID mappings in the diff.
- Define every op in a versioned catalog. Its schema includes required fields, enums, ranges, units, defaults, target types, scope, dependencies, reversibility, and context. Do not bypass type validation with generic JSON patch.
- Declare and validate runtimes for scripts / extensions. Do not fill gaps in the custom engine by making arbitrary bpy calls.

### Operation Families

Implement all families below in the first release. Each variant MUST cover every type in the fixed Blender profile and feature catalog.

| Family | Operations |
| --- | --- |
| Scene / Collection / Object / Data | `scene.*`, `collection.*`, `node.*`, `data.*`: create, update, duplicate, delete, link / unlink, parent, origin, convert, make_single_user |
| mesh | `mesh.*`: element selection, position / attribute editing, extrude, inset, bevel, loop_cut, bridge, weld, split, dissolve, fill, subdivide, triangulate, knife, bisect, remesh, mirror |
| Modifier / graph | `modifier.*`: create / update / reorder / apply all types. `graph.*`: all nodes, sockets, links, groups, interfaces, zones, attributes |
| sculpt / paint | `sculpt.*`, `paint.*`: brush / stroke, mask, face set, symmetry, Dyntopo, Multires, texture / vertex / weight |
| UV / material | `uv.*`: unwrap, pin, pack, transform. `material.*`: all shaders and slots. `image.*`: pixels, UDIM, color space |
| rig / animation | `bone.*`, `pose.*`, `constraint.*`, `driver.*`, `shape_key.*`, `action.*`, `keyframe.*`, `fcurve.*`, `nla.*` |
| physics | `physics.*`, `simulation.*`: system, solver, collision, force, seed, initial state, cache conditions |
| Grease Pencil | `grease_pencil.*`: layer, stroke, fill, frame, attribute, effect, animation |
| render / composite | `camera.*`, `light.*`, `world.*`, `render.*`, `color.*`, `compositor.*`: all settings, graphs, passes, outputs |
| tracking / video | `tracking.*`, `mask.*`, `sequencer.*`: track, solve, spline, strip, retiming, effect, sound |
| asset / library | `resource.*`, `library.*`, `asset.*`: retrieval, reference updates, pack / unpack, link / append, override, metadata |
| script / extension | `text.*`, `property.*`, `extension.*`: source, custom data, dependency registration, permitted native adapters |

### Basic Operations

- node.create: id and kind are required. A primitive requires params or a reference to the corresponding data. A group uses Empty as a convenience input. If omitted, it belongs to the active Scene's root Collection.
- Default transform: translation=[0,0,0], rotation=[0,0,0,1], scale=[1,1,1]. name=ID, tags=[], parent=null. Visible, renderable, and selectable.
- node.update: target and non-empty set are required. params and transform update fields. Changes to shared geometry use the scope contract.
- node.duplicate: a new ID is required. `mode: independent|linked` defaults to independent. `recursive:true` duplicates the subtree and returns ID mappings. Duplicate groups, bones, and instances according to their target type as well.
- node.parent: specify parent and `keep_world`, default true. Preserve parent inverse and shear to maintain appearance. Do not decompose into simple TRS without authorization.
- node.delete: if children exist, specify `recursive:true` or `reparent:"to_parent"|"to_root"`. `to_parent` moves children to the nearest parent not being deleted; `to_root` moves them to no parent. Both preserve the world transform with `keep_world` (default true). `recursive:true` and `reparent` MUST NOT be used together. Explicitly specify cascade / unlink handling for referenced data. Determine whether deletion is allowed from the structure at the start of the operation.
- material.create / update support both simplified PBR input and full shader graphs. For a referenced material.delete, explicitly handle assignments; do not silently change them to the default.
- An ID collision yields ID_EXISTS; do not rename automatically. For Blender name collisions, also create a name mapping report and manage by ID.
- A stroke is a sequence of world-space samples with time, pressure, radius, strength, falloff, symmetry, and seed. Convert screen input from the Snapshot, camera, and depth, and record the resulting world-space sequence.
- Validate attribute, material, UV, weight, shape key, and rig mappings after topology changes. If preservation is impossible, return an error; do not save corrupted creation data based on guesses.

## 6. Atomic Editing and Retries

init: Create the source of truth in a nonexistent or empty directory. Parent directories MAY be created. A non-empty directory yields OUTPUT_EXISTS. result contains created=true and scene_file.

apply / import / undo / redo:

1. Validate input schema and runtimes. Acquire a non-blocking OS exclusive lock; if unavailable, return SCENE_BUSY.
2. Read the source-of-truth Snapshot. Require a matching base_revision; otherwise return REVISION_CONFLICT.
3. Apply operations in array order to a working graph, or incorporate the imported structure, assets, libraries, and compatibility payloads.
4. Validate all structure, dependencies, evaluation in the target context, and compatibility conditions. Reject missing dependencies or losses according to policy.
5. Complete new referenced blobs and assets in a temporary area. With `--preview`, complete all images and the manifest as well.
6. Place immutable blobs and derived data at dedicated paths. Sync the new scene.json and commit by atomic OS replacement.
7. Release the lock. Return the revision, diff, ID mappings, loss / migration report, and output path.

- If an operation, evaluation, import, asset, preview, or I/O fails before commit, preserve the old source of truth and revision. No partial application.
- New blobs not referenced by the source of truth are not considered committed even if left behind after interruption. They MAY be removed by garbage collection that protects references from active Snapshots.
- Increment the revision by 1 only if the final normalized content changes. Do not increment for operations=[] or a no-op.
- apply requires preview when size is specified. apply writes only to an internal dedicated path; no out parameter.
- dry-run validates targets, diffs, compatibility / dependencies / losses. It does not change the source of truth, revision, or external outputs. Combining preview and size is INVALID_ARGUMENT.
- If disconnected after commit but before receiving the response, the new revision may already be saved. Check with inspect. Do not apply twice against the same base_revision.
- Locks are OS-managed and released on exit. Do not determine busy status from the presence of a lock file. Guarantees apply to local filesystems supporting atomic replacement and OS locks.
- Observations, export, render, and bake pin one source-of-truth Snapshot and its asset hashes. Do not mix them with another apply during processing.
- Generate simulation / geometry / texture / animation cache keys from Snapshot inputs, profile, solver, seed, frame, and execution policy. Do not use an old cache when conditions change.
- Store successful commits in immutable history. History lists revisions, operations, diffs, before / after hashes, and restorability. undo / redo restores the creation state from history and commits a new revision; it does not roll back to a past revision number.
- steps is a positive integer, default 1. Insufficient history yields TARGET_NOT_FOUND. Record new edits after undo as a separate branch and disable automatic redo of the previous branch. If original assets or external dependencies cannot be restored, fail without changes.
- redo / undo results also return committed, changed, base_revision, candidate_revision, changes, and the restored history ID. After a crash, restore only completed commits as the source of truth; do not automatically apply an uncommitted in-progress operation.

apply result: committed, changed, base_revision, candidate_revision, changes, operations, id_mappings, previews. changes lists created / updated / deleted by registry in ID order; operations are in input index order. For dry-run, committed=false and the common scene fields refer to the original revision.

## 7. Inspect and Assets

- With no id, return all Objects in the target Scene; with a tag, return all matches. Zero matches yields TARGET_NOT_FOUND. An id identifies the type and registry and can also retrieve Data-Blocks, etc.
- Do not implicitly add descendants. Show structure through parent, children, Collection membership, data references, and instance paths.
- Retrieve the full Project inventory outside the filter from the summary. Include hidden and unused items in counts.
- result: summary, items, references, resources, features. items contain typed information for the specified registries. Also return convenience values such as node params.
- If evaluation of an Object in targeted inspect (`--id` / `--tag`) yields `UNSUPPORTED_FEATURE`, the failure response `error.code` MUST be `UNSUPPORTED_FEATURE` and include `error.details.feature_id`.
- Inspect of all Scenes (without `--id` / `--tag`) MUST NOT be interrupted by an individual Object's `UNSUPPORTED_FEATURE`. Keep that Object in `result.items` and record `code`=`UNSUPPORTED_FEATURE` and `details.feature_id` in its `evaluation_error`. Return other items and the summary as well; do not return unevaluated values as evaluated.
- Object: id, name, kind, tags, parent / inverse, Collection, data / sharing, slots, Modifiers, Constraints, drivers, visible / render / selectability, raw / evaluated transforms and bounds / dimensions.
- A node's `transform.evaluated` is its local value after frame evaluation: translation, rotation `[x,y,z,w]`, scale, and local `T × R × S` matrix. Obtain the final placement including hierarchy / parent inverse / Constraints from `transform.world`. world always returns a column-major 16-element matrix and `decomposable`; if decomposable, also return translation, normalized / canonicalized rotation, and scale. For shear or a singular matrix, set `decomposable:false` and translation / rotation / scale to null while retaining the matrix. For negative scale, preserve glam's `to_scale_rotation_translation` convention and put the negative sign on the X scale if the determinant is negative.
- Data-Block: source / evaluated geometry, counts of vertices, edges, polygons, triangles, corners, and attributes, UVs, weights, shape keys, and sharers. Bones, graphs, Actions, physics, etc. use typed properties and references.
- Evaluation Snapshot: Scene, View Layer, frame, engine version, and evaluation hash. Bounds are `{"min":[x,y,z],"max":[x,y,z]}`; empty bounds are null.
- Group / Collection bounds aggregate member geometry in the selected Scene. Specify counts and target visibility. Distinguish data counts for shared geometry from displayed instance counts.
- `--features` lists create / edit / evaluate / render / import / export / preserve_only, required dependencies, and reasons for gaps. Do not hide gaps.
- Assets return URI, hash, kind, owner, packed / external, missing / changed status, library relationships, and restoration conditions. check verifies actual files against their hashes.
- Do not incorporate external asset changes into the source of truth through observation alone. Use apply / import to update the URI and hash.

## 8. Preview, Render, and Bake

### Preview Views and Modes

Default views=iso, mode=solid. Presets are orthographic; when a camera is specified, use all projection settings of the saved camera.

| View | Camera position / direction | Up direction on screen |
| --- | --- | --- |
| front | [0,-1,0] | +Z |
| back | [0,1,0] | +Z |
| right | [1,0,0] | +Z |
| left | [-1,0,0] | +Z |
| top | [0,0,1] | +Y |
| bottom | [0,0,-1] | +Y |
| iso | normalize([1,-1,1]) | +Z projected onto the screen |

- The target is the center of displayed geometry, instances, and volume bounds in the selected context. Set ortho_height to 1.1 times the largest side of the projected bounds.
- For an empty scene, target=[0,0,0] and ortho_height=1m. Cameras, lights, and bones without geometry can also be shown as overlays / selections.
- Modes: solid, beauty, wire, normal, depth, id. solid is for checking shape; beauty evaluates shaders, textures, lighting, volumes, and color management. Do not silently approximate PBR settings with Lambert.
- Fixed solid background sRGB=[242,242,242,255] and fixed studio lighting. beauty uses the Scene's world / light / render settings.
- Treat scalar density in VDB / inline volumes as an extinction coefficient [m^-1]. density=1 means 1/m. Calculate transmittance with the Beer-Lambert equation, `T = exp(-density × distance[m])`. Convert ray distance using `unit.scale_length` (meters per scene unit), and sample after applying the Object world transform and VDB grid transform.
- Specify AA, samples, seed, transparent background, overlay, engine / device, and OCIO settings in the manifest. Distinguish sampling policies for color and selection passes.
- The default preview context is the viewport selection context. render uses render visibility and the View Layer. Evaluate exclusions, holdouts, and instances.
- Preview all display targets that can be evaluated in the selected context. If an individual Object cannot be evaluated due to `UNSUPPORTED_FEATURE`, do not interrupt the entire Preview: omit that Object from rendering and add a warning for each affected target to `warnings`. Each warning MUST include `code`=`UNSUPPORTED_FEATURE`, identifying information for the target Object, and `feature_id`. Do not substitute approximate or unevaluated geometry.
- The ID / depth selection pass uses one center sample with no AA. Apply material conditions for opaque / cutout. For multiple transmission / volume hits, generate a hit set with provenance; the default pick returns the nearest selectable surface.
- If the center ray hits the background at a visual AA boundary pixel, hit=false. The visible color after transmission may differ from the selected surface. Record the selection rule in manifest pick_policy.
- Resolve equal depths by node ID, instance path, then element ID.

### Preview Output and Manifest

```text
previews/
  iso.png
  iso.ids.bin
  iso.depth.bin
  iso.elements.bin
  iso.manifest.json
```

- If out is omitted: `<scene>/.potter/previews/r<revision>/<render_key_hex>/`. The hash path is 64 hex characters without the sha256: prefix.
- Record scene ID, revision, scene_hash, evaluation_hash, engine, profile, frame, View Layer, camera, units, settings, and pick_policy in each view's manifest.
- The camera consists of projection, position, target, up, near / far, and matrix, with typed storage for specific values such as orthographic / perspective / panorama, lens / stereo, etc.
- render_key is the JCS SHA-256 of the above and width / height and referenced asset / cache hashes. Exclude paths and timestamps.
- Images have path, width, height, format, and file hash. Buffers also have path, type, dimensions, and hash. Paths are relative to the manifest; absolute paths, `..`, and symlinks outside the directory are not allowed.
- object_id is row-major little-endian uint32, with 0 for background. Resolve through the ID-ordered objects mapping to node, data, instance path, and origin references.
- depth is row-major little-endian float32, camera-forward distance in m, with 0 for background. Record projection-specific ray-depth definitions for panorama, etc.
- elements is the evaluation element mapping index for the center sample, little-endian uint32, with 0 for unsupported / background. Store mappings for persistent elements, source elements, and generated provenance.
- Each base buffer is width×height×4 bytes. Large mappings use hash-referenced sidecars. Even for frame animation, the manifest is pinned to the context of one image.
- Generate all views temporarily before publishing, and atomically replace each view's manifest last. Atomic replacement of the entire external directory is not guaranteed; reject incomplete sets by hash.
- Reuse an existing set if the render_key matches and all files are valid. If an existing set differs or is incomplete, return OUTPUT_EXISTS unless overwrite is specified.
- Do not change the source of truth or revision. result.previews returns absolute paths, context, and hashes for all generated files in the requested view order.

### Render and Bake

- render uses the saved camera, world, light, shader, pass, compositor, and Scene output settings. Without a camera, return TARGET_NOT_FOUND. The default is the CPU path engine; realtime / GPU require explicit selection.
- render supports stills, image sequences, video, audio, and VSE output. Validate formats such as PNG, JPEG, TIFF, OpenEXR, MP4, WebM, WAV / FLAC, etc., and codec / bit depth / alpha in the schema.
- Render output returns per-frame hashes, frame / subframe, samples / seed, fps, OCIO, AOV / pass, and a common run manifest.
- Image sequence filenames use the shortest round-trip decimal representation of the frame number. For fixed-point notation, zero-pad the signed integer part to at least 4 digits (e.g. `frame_0012.5.png`); retain exponent notation. Distinguish repeated frames with `__2`, `__3`, etc., and match the manifest paths.
- bake targets simulation / texture / geometry / animation. Validate solver, UV, target, frame, cache / output conditions, and return outputs and dependency keys.
- Neither command changes the source of truth or revision. Explicitly import / attach generated assets to the source of truth with apply.
- Floating-point results may differ across devices / solvers / parallel execution even for the same Snapshot. Reproduce with a specified environment and seed; validate numeric errors against feature-specific fixtures.
- Do not make missing custom feature implementations appear successful by forwarding to Blender render / bake. render / bake require evaluation of all targets; fail processing with `UNSUPPORTED_FEATURE` for an individual Object and return `error.details.feature_id` and `error.details.node_id`. Do not publish partial output as success.

## 9. Pick

- pixel is an integer in the original PNG dimensions, with origin at top left, x to the right, and y down. Its center is (x+0.5,y+0.5); valid range is 0<=x<width and 0<=y<height.
- The caller converts coordinates from a downscaled image to the original dimensions. No automatic scaling.
- Validate the manifest's scene ID, revision, hash, evaluation context, engine version, and dependency hashes. A mismatch yields STALE_RENDER.
- Validate file hashes, buffer dimensions, mappings, camera, and projection. Invalid data yields RENDER_INVALID; missing data yields FILE_NOT_FOUND.
- Default domain is object. For vertex / edge / face / bone / stroke / point, use the corresponding selection channel, ray, and distance threshold. Missing required channels yields INVALID_ARGUMENT; do not incorrectly substitute an Object.
- Verify the center ID against ray evaluation using the same camera. Depth tolerance is max(1e-6m, abs(depth)×1e-6). World position / normal are f64 evaluation results.
- result: hit, pixel, domain, target, item, instance_path, source, editable, world_position, world_normal, depth, evaluation_index, snapshot_hash.
- On a hit, target is an ID / element selector usable by operations, plus a required snapshot_hash. For elements from generated / library sources that cannot be edited directly, return editable=false, source, and the reason.
- The corresponding value is null for targets without geometry or a surface normal. Volumes return entry / exit and ray information. A world background without an Object is hit=false.
- For background, only pixel, domain, and snapshot_hash are non-null; exit code is 0. Do not use evaluation indices as persistent editing IDs.
- Hits MAY be added for transmission / overlap candidates at the same pixel. The primary target follows manifest pick_policy.

## 10. Validate

Check the source of truth, referenced blobs, Scene context, dependencies, all features, and the specified output format without changes.

| Check | Result |
| --- | --- |
| Schema, IDs, references, parent / Collection cycles, invalid indices, non-finite values | error |
| Corrupt / missing assets, changed hashes, missing library / extension / runtime | error |
| Rig / skin / shape key mapping, graph links / socket types, driver / Constraint dependencies | error |
| Frame / fps, Action / NLA, simulation / cache, shader / color / compositor / VSE inconsistencies | error |
| Unimplemented standard features, features not representable in the output format | error; report only if lossy conversion is explicitly permitted |
| Non-manifold geometry, inconsistent winding, zero area, singular transform, open boundaries | warning. Preserve source data; do not force solidification |
| Unused Data-Blocks, unreferenced elements | information. Preserve as Blender creation data |

- Also provide self-intersection, interference, and wall-thickness checks using native solvers / geometry. Specify targets, tolerance, and volume / surface interpretation. Include applicability to open / curve / volume data, etc. in the report.
- Do not pass "check not implemented"; return not_supported with a reason in checks. Insufficient evaluation of standard features fails first-release acceptance.
- Warnings do not prevent normal saving, rendering, or export. Exclude zero-area elements from rendering; do not delete creation data without authorization.
- result contains valid, issues, checks, summary, compatibility, losses. An issue contains severity, code, message, data_id, pointer, and details.
- valid=true if there are no errors. In strict mode, warnings also cause exit code 4 / ok=false / VALIDATION_FAILED; preserve valid's error determination.
- Return available reports even on error. If data cannot be decoded, result=null. Information alone does not fail strict mode.
- apply / import / export also use the same structure, dependency, feature, and format checks. Specify coverage for unevaluated frames.

## 11. Blender Import and Blend Adapter

### Import

- format is required and MUST match the source extension. mode defaults to append; replace replaces the entire creation graph. Preserve the working scene_id. Import all Scenes and hidden / unused Data-Blocks.
- base_revision is required. Operate only on an initialized Scene. Convert and validate assets and losses, then commit atomically as described in Section 6.
- .blend default is an editable import. Do not flatten / bake Modifiers, nodes, rigs, animation, physics, etc.
- Reuse IDs if they are stored in the .blend. Issue typed IDs for data not previously assigned IDs and save mappings. Do not determine identity by name sanitization alone.
- Explicitly remap colliding IDs in append to an import namespace, and update sharing, links, and driver references as well. Do not overwrite existing IDs. For replace reimports, use existing mappings and persistent IDs to obtain diffs.
- asset-policy defaults to copy. Resolve dependencies relative to the source .blend and library locations; register packed status, source URI, hash, and unknown payloads. For link, specify external restoration conditions.
- Preserve library overrides, shared data, and Data-Block user relationships. Do not implicitly decompose, make single-user, or approximate materials.
- result: format, source, committed, changed, base_revision, candidate_revision, changes, id_mappings, resources, compatibility, migrations, losses.

### Adapter Boundary

- Launch the fixed-version Blender child process only for .blend binary reading / writing. Do not depend on a GUI connection, MCP add-on, or a running Blender instance.
- The bridge extracts / reconstructs complete creation data and references in a typed intermediate format. Do not ask Blender to evaluate geometry, apply Modifiers, render, or simulate.
- Use the saved source .blend / unknown payloads as preservation inputs when reconstructing exports. The potter graph is the source of truth for known features; do not discard edits by re-exporting only the old source file.
- Find the executable in this order: --blender -> POTTER_BLENDER setting -> PATH. If not found, return BLENDER_NOT_FOUND; if the version mismatches, return BLENDER_VERSION_UNSUPPORTED.
- Run subprocesses in background mode with factory settings and automatic file script execution disabled. Run only the fixed internal I/O script. Do not launch a Blender child for normal editing / rendering.
- Capture bridge stdout / stderr. Validate intermediate files and reports before importing into the source of truth. Do not treat the exit code alone as proof of successful conversion.
- If an extension is required for reading / writing, resolve it using the declared profile. Fail with a report if data cannot be preserved.
- Provide and validate compatible Headless Blender binaries for each supported OS / architecture. Distinguish normal core availability from .blend adapter availability in capabilities.

See the [Blender Compatibility requirements](blender-compatibility.md#3-meaning-of-blend-compatibility) for full-feature round trips, version migrations, asset preservation, and unknown-data policy. Blender's own compatibility constraints: [Blend File Compatibility](https://developer.blender.org/docs/handbook/guidelines/compatibility_handling_for_blend_files/).

## 12. Export and Format Compatibility

| Format | Import | Export | Main contents |
| --- | --- | --- | --- |
| blend | REQUIRED | REQUIRED | All Scenes, creation graph, hidden / unused data, editing information, sharing, library, script / UI information preservation |
| glb / gltf | REQUIRED | REQUIRED | Mesh, material / texture, camera / light, skin, morph, animation, metadata, supported extensions |
| usd / usda / usdc / usdz | REQUIRED | REQUIRED | Scene, instances, materials, geometry, animation, volume, assets |
| abc | REQUIRED | REQUIRED | Alembic geometry / transform / attribute time-series cache |
| fbx | REQUIRED | REQUIRED | Hierarchy, mesh, material, rig, animation. Report variants / losses |
| obj / ply / stl | REQUIRED | REQUIRED | Geometry, normal / UV / color / material as supported by the format |
| bvh | REQUIRED | REQUIRED | Skeleton, motion |
| svg | REQUIRED | REQUIRED | Curve / Grease Pencil. Specify projection, frame, sampling |
| pdf | — | REQUIRED | Grease Pencil projection, frame, page |
| PNG / JPEG / TIFF / EXR / video / audio | Asset import | Render / bake | Color, bit depth, alpha, codec, frame / fps / audio sync |

- Specify each format's representation scope in the formats catalog. Do not claim preservation of creation features that do not exist in the target format.
- .blend is a round-trip format for the entire creation project. GLB, etc. are interchange formats for assets / geometry / motion, etc. as defined by the formats catalog. Determine the preservation scope for the selected format in advance.
- Record in conversions for that format contract when an interchange format does not include the creation graph, when generating evaluated geometry for primitives / Modifiers, or when triangulating / converting bases / quantizing to float32. The basic box-to-GLB workflow does not require allow-lossy.
- Reject by default conversions that lose targeted geometry, materials, rig, motion, etc. Include feature_id, data_id, reason, and alternatives in losses. Permit only explicitly specified sampling / shader approximation / omission, etc. with `--allow-lossy`. Do not implicitly flatten a .blend creation graph.
- Non-`.blend` export requires evaluation of all targets in the selected context. If an Object cannot be evaluated, fail with `UNSUPPORTED_FEATURE` and return `error.details.feature_id` and `error.details.node_id`.
- .blend export reconstructs all creation data rather than converting the selected frame to mesh output. Preserve hidden / unused data and Action / NLA / physics settings for all frames.
- Evaluated exports such as GLB use render visibility and required ancestors in context. Preserve skin / morph / animation / texture as exportable structures.
- Convert potter Z-up world(x,y,z) to glTF(x,z,-y). Convert transforms, normals, bones, bind matrices, and animation in the same basis; retain m. [glTF 2.0 Specification](https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html)
- .blend restores Z-up, original Scene units, rotation mode, parent inverse, and shared data. Write materials, Modifiers, nodes, and rigs as creation data.
- Store potter persistent IDs, tags, element IDs, and Scene hash / revision in .blend custom properties / attributes, etc. and format metadata. Use a namespace that does not conflict with original user properties.
- With pack specified, pack assets the format can contain. Put data that cannot be packed, such as caches / libraries, in a relative sidecar bundle plus manifest, and return the list of generated files.
- GLB embeds binary data / supported images. For glTF, etc., stage the entire sidecar file set and verify dependency hashes. Distinguish a format's "single file" from a bundle.
- The output parent directory MAY be created. If output exists, return OUTPUT_EXISTS unless overwrite is specified. Publish after validating temporary output. Use atomic replacement for a single file; switch a completed directory or commit a manifest for a bundle.
- Do not change the source of truth or revision. Preserve existing output on failure before commit. Do not silently discard unknown data that cannot be preserved during export, even when lossy is specified.
- result: format, path, files, hash, bytes, counts, context, dependencies, compatibility, conversions, losses.

## 13. JSON Responses, Compatibility, and Limits

Common format:

```json
{
  "schema_version": 1,
  "command": "apply",
  "ok": false,
  "scene": null,
  "result": null,
  "warnings": [],
  "error": {
    "code": "INVALID_OPERATION",
    "message": "parameter is outside its valid range",
    "details": {
      "operation_index": 0,
      "pointer": "/operations/0/params/size"
    }
  }
}
```

- On success, ok=true and error=null; result contains the command-specific result. On failure, error is required and any available report remains in result.
- scene contains id, absolute path, revision, and hash. Return it on failure too if already retrieved. It is null if not retrieved / for schema retrieval.
- command is the subcommand name; it is null for CLI errors before command determination. warnings is always an array, with code, message, data_id, and details.
- Results involving evaluation specify snapshot_hash, Scene, View Layer, frame, engine / profile, and execution policy.
- error.code is machine-readable, message is brief English, and details contain JSON Pointer, zero-based operation_index, target ID, feature_id, dependencies, etc.
- apply stops at the first fatal error. validate / import compatibility checks list all problems that can be retrieved.
- The schema result contains kind, op, schema; capabilities / formats are typed catalogs. All use the same response envelope.
- schema_version is independent for the source of truth, operations, manifest, and response. Profile, engine, bridge, and solver versions are also independent.
- Reject unknown input fields. Callers MAY ignore optional fields added to output. Increment the corresponding schema_version for breaking changes.
- When updating the Blender baseline version, validate feature / RNA catalog differences, migrations, and round-trip fixtures. Do not automatically claim compatibility with future versions.

### Limits and Reproducibility

- Default control JSON limit is 64 MiB / 10,000 operations per batch. Use typed binary / chunk references for large arrays, strokes, images, volumes, and caches.
- Default budget: 1,000,000 source-of-truth entities, 100,000,000 evaluated triangles, 8 GiB memory; configure CPU thread count. Config can change these values. Exceeding a limit yields LIMIT_EXCEEDED; no partial saves.
- Distinguish creation feature limits from resource budgets. Do not treat large Blender Scenes as unsupported features due to a fixed 1,000,000-triangle limit. Implement chunk / tile / stream processing.
- Process one Preview at a time; do not hold all views' buffers simultaneously. Process video and caches by frame / chunk as well.
- Source-of-truth JCS hashes, references, and persistent IDs are OS-independent. Reproduce ID mappings and center-selection buffers exactly with the same engine version, profile, input, evaluation context, and runtime environment.
- Record engine / solver / device, seed, thread, OCIO, etc. for beauty / simulation / bake. Validate numeric errors per fixture. Do not require byte-for-byte identity for .blend due to compression, save-time conversion, or timestamps.
- Do not confuse Blender byte differences with compatibility gaps; judge by editable semantics, references, and evaluation results.

## 14. Implementation Responsibilities and First-Release Acceptance

| Unit | Responsibilities |
| --- | --- |
| CLI / schema | All commands, catalog, arguments, types, context, responses, exit codes |
| model / store | All Data-Blocks, Scenes, sharing, references, assets / binaries, hashes, Snapshots, history / undo / redo, atomic commit |
| operations / selection | All typed ops, scope, element IDs, strokes, change mappings, undo data |
| geometry / graph | All shapes, topology, attributes, all Modifier and Geometry Nodes evaluation, dependency graph evaluation |
| sculpt / paint / UV | Brush, sample, remesh, Multires, texture / vertex / weight, UV / UDIM |
| rig / animation | Bone, skin, pose, Constraint, driver, shape key, Action / NLA |
| physics / cache | All solvers, collision, force, simulation nodes, bake, reproducibility, invalidation |
| renderer / query | CPU / GPU, PBR, path / realtime, volume, hair, pass, color management, pick |
| compositor / media | All nodes, tracking / mask, VSE, image / movie / audio, codec |
| library / asset | Link / Append / Override, dependency retrieval, pack, hash, catalog |
| exchange / blend | All format conversions, Headless .blend I/O-only bridge, preservation, loss, version checks |
| compatibility tests | Blender standard full-feature catalog / fixtures, native evaluation, editable round-trip comparison |

Do not consider the first release complete until all responsibilities are implemented. Internal work dependencies are not release phases that defer features.

Blender evaluation MAY be used in tests to generate / compare compatibility fixtures. Blender MUST NOT be used for normal product creation, evaluation, or rendering.

Acceptance criteria:

1. Pass every category and fixture in the [Blender Compatibility requirements](blender-compatibility.md). No unimplemented standard types / Modifiers / Constraints / nodes / sockets.
2. Complete the basic workflow in Section 2 without a GUI. Normal creation / rendering / baking do not launch Blender; only .blend import/export uses the Headless Blend Adapter.
3. After the first batch, revision=1, body world bounds=[-0.5,-0.3,0] to [0.5,0.3,0.8], dimensions=[1,0.6,0.8]. Correctly evaluate params and rotation in the update example as well.
4. Preserve the old source of truth and revision after invalid ops / assets / evaluation / preview / import or pre-commit I/O failure. dry-run has the same final diff but does not save.
5. At most one conflicting change against the same base_revision succeeds. Observations and outputs MUST NOT mix with another commit during Snapshot processing.
6. Do not confuse fixed IDs, duplicate names, tags, many, sharing scope, subtree duplication, or bone / element targets. Preserve mappings after reimport and Blender rename.
7. Validate all Scenes, View Layers, frames / subframes, units, negative / singular transforms, parent inverse, and shared geometry.
8. Verify consistency among PNG, ID / depth / element, camera, and frame. Distinguish pick hit / background / transparency / volume / instance / generated provenance / stale.
9. Distinguish all validation categories, format losses, and missing assets / runtimes. Do not silently flatten a too-new .blend or unknown extension.
10. Round-trip .blend while preserving Modifiers / graphs / rigs / Actions / physics / compositor / VSE, hidden / unused data, libraries, and packed assets. No semantic differences outside explicitly targeted data.
11. Verify exports for all interchange formats and evaluate after reimport. For GLB, check skin / morph / animation / texture, axes, and bind matrices.
12. Reproduce brushes / solvers / rendering with the same seed and environment. All JSON can be parsed in one pass; child Blender logs do not mix into stdout.
13. undo / redo atomically restores creation graph, sharing, assets, animation, etc., and increments the revision. Do not confuse history and the source of truth across restarts, branches, missing assets, conflicts, or crashes.

Do not release as supporting all features while implementation or acceptance is incomplete.
