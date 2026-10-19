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
pot export ./scene --format vrm --out character.vrm --json
pot validate ./scene --format vrm0 --strict --json
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
- `pot validate --format vrm|vrm0` runs the same representability checks as export without writing a file; failures report `VRM_EXPORT_PREFLIGHT_FAILED`.
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
- Revisions range from 0 to `2^53 - 1`. scene_id is a UUID v4 fixed for the lifetime of the working directory. The scene JSON format uses schema_version=4. Version 1 documents are upgraded on load through version 2 (`model/migrate.rs`: `uv_map` and `blender_uv_layers` become the `uv_layers` store, and texture `uv_map` becomes `uv_layer`); version 2 documents are upgraded to version 3 (legacy Blender shading and mesh attributes become native mesh data); version 3 documents are upgraded to version 4, where an EXR image labelled straight becomes premultiplied, because Blender and the loader read an EXR as associated alpha. That step reads the image's blob or linked source file from the project root, so every parser takes the root; an image whose bytes cannot be read keeps its label, and the loader reports it when used. History records are checked against their content hash before they are upgraded, and upgraded records are not rewritten on disk. Saves write version 4 only, and no other version is accepted.

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
- The mesh source of truth consists of vertices, edges, polygons, and corner attributes. Preserve triangles / quads / n-gons and loose vertices / edges. Triangulate when observing, rendering, or converting to a format that requires it, with Blender's looptris (`Mesh::tessellate`, as `mesh_tessellate.cc` computes them). It never fails on self-intersecting or zero-area faces, so consumers that read looptris get Blender's diagonals.
- UVs, normals, colors, weights, creases, seams, sharpness, and material indices are attributes on their respective domains. Vertex splitting for rendering MUST NOT change source-of-truth connectivity.
- Mesh data is stored once as ID-keyed attributes on point, edge, face, or corner domains. UVs use the `uv_layers` store. Color values use named attributes of type `color`, with POINT values keyed by `v<id>` and CORNER values keyed by `f<id>` in face-corner order; `color_layers` stores layer order plus the active and render names. User-named color layers and generic attributes use the injective `storage_key(name)` codec, while registry layer names remain user names. Canonical RGBA values are linear and are not clamped, including HDR floats. Blender's source FLOAT_COLOR/BYTE_COLOR encoding is metadata only. Generic native attributes use `float`, `int`, `float2`, `float3`, `bool`, `color`, or `quaternion`; Blender-only state remains in compatibility data without duplicating live geometry, shading, UV, color, or generic values.
- Mesh-rebuilding modifiers report each output vertex's input vertices (with interpolation weights) and each output polygon's input polygon. Vertex groups, edge creases, native attributes, skin data, colors, UV layers, and custom-normal code state follow those origins under per-modifier rules. Shading uses native `shade_smooth` (face bool), `sharp`/`seam` (edge bool), and `custom_normal` (`normal_code` per corner: `null` Auto/code0, `[i16,i16]` authored Blender fan code, `"explicit_zero"` for an unrepresentable legacy literal). Optional `corner_order` stores loop ordinals that define code fan spaces when Blender order differs from mesh storage, as part of the same normal state. `custom_normal.values` present (including all-Auto rows) means a Blender custom layer exists; values absent with retained `corner_order` preserves only the basis for later normal authoring and does not allocate a custom layer. The public normal setter preserves a literal zero as `explicit_zero`; Subdivision's already-normalized setter includes zero vectors in the fan average but stores those corners as Auto. Corner vectors are derived at evaluated positions. Blender-only metadata is not a second geometry or shading store. A modifier that cannot carry an attribute reports `modifier.attribute_dropped` as a warning on `modifier.apply`, `pot inspect`, `pot export` of evaluated formats, and `pot preview`.
- UV layers (the `uv_layers` store) follow the mesh-rebuilding origins under Blender's per-modifier UV rules: subdivision interpolates each layer as a face-varying limit stencil for every `uv_smooth`, boundary, and crease rule, with per-corner pins interpolated by positive weight; UV-space transforms (array offset, mirror, screw, UV Project, UV Warp) compute in Blender's V-up space; remesh and skin drop them.
- Subdivision (levels > 0) with Use Custom Normals off writes no custom-normal values or layer when output is stored in Blender's loop order; later authoring allocates values in that same order. With Use Custom Normals on, output codes are re-encoded through the shared setter in already-normalised mode. The evaluated vertex, edge, face and loop arrays follow Blender's `subdiv_foreach` offsets and callbacks: coarse vertices, edge samples in input-edge order, face-grid vertices in coarse-face order, boundary edges before internal face edges, then faces and loops in emission order. Each face's loops keep Blender's callback rotation, so the resulting custom-normal basis is identity; imported non-identity input `corner_order` is honoured when sampling. UVs, pins, colors, generic attributes, materials, flags and vertex groups follow the same permutations. Codes match Blender with alpha exact, beta within one step, and Auto exact across the 14 code-carrying cases (3,344 loops). The cross-platform variance measured on the original 147-case corpus is 11 of 2,656 codes differing by one beta step between linux-x64 and macOS arm64, never in alpha; see `tests/common/blender_code_tolerance.rs`.
- Subdivision mixes face-sampled elements (vertices, loops, and byte colours) stage by stage, as Blender does: each of the four ptex slots mixes its sources first, then the slots mix by their weights. Coarse corners copy their value. A loose-edge vertex is one two-source `mix2` of its edge's ends, with no outer mix. A two-source stage (an n-gon slot's edge midpoint) is also `mix2`; a stage of more sources (the n-gon centre) and the outer slot mix are weighted `mix_indices`. FLOAT, FLOAT_COLOR, and BOOLEAN data are linear, so the staging does not change them. INT data rounds per slot and again per outer mix (`mix2`/`mix4`, `BKE_attribute_math.hh`). QUATERNION data uses two mixes: a weighted mix is the weight-normalised mean of exponential maps, mapped back (`mix_indices`/`mix4` with `quat_to_expmap`/`expmap_to_quat`; `QuaternionBase::expmap`, `BLI_math_quaternion.hh`), and a two-source `mix2` is spherical interpolation along the shorter arc (`math::interpolate`, `a * pow(a^-1 * b, t)`). BYTE_COLOR rounds each channel of a two-source `mix2`, and truncates a weighted sum to a byte (`mix_indices`), so Potter's linear colour store is re-encoded from the same bytes Blender interpolates.
- Generic `quaternion` attribute values use the native `[x,y,z,w]` order, the same as this spec's quaternion order. Blender stores `w,x,y,z`; the `.blend` bridge converts both ways (`_quaternion_to_native`, `_quaternion_from_native`), and an element with no value takes the identity `[0,0,0,1]`.
- UV layers are the `uv_layers` mesh attribute: an ordered list of named layers. Each layer holds per-face corner `[u, v]` values keyed by persistent face ID, with optional per-corner `pinned` flags. `active` is the edit target for `uv.*` ops and `render` is the render and export default; both name a layer, or are null when the mesh has none. A mesh holds at most 8 layers, Blender's limit. Values use glTF's V convention (v down); `.blend` import and export convert to and from Blender's V-up convention. A texture reference (`TextureRef.uv_layer`, MToon textures, shader `ShaderNodeUVMap`) names a layer, and `null` names the render layer. A name no layer has falls back to the render layer and is reported as `uv.layer_missing`: a warning in render results, a loss on export. `uv.layer_create` without a name yields `UVMap`, `UVMap.001`, and so on; a taken name is `uv.layer_exists`. `uv.layer_delete` of a layer that a texture or modifier still names is `uv.layer_in_use`. `uv.layer_rename` updates those references. `uv.layer_set_active` and `uv.layer_set_render` set the two roles. `uv.unwrap`, `uv.transform`, `uv.pin`, `uv.pack`, and `paint.texture` take `uv_layer` (default: the active layer for `uv.*`, the render layer for `paint.texture`); ops work in glTF's V convention, and modifier UV-space math works in Blender's.
- Decimate follows Blender 5.2.2 `bmesh_decimate_collapse` / `bmesh_decimate_unsubdivide` / `bmesh_decimate_dissolve` on a BMesh-shaped topology: COLLAPSE keeps Blender's heap order, float32 costs, disk and radial order, symmetry pairing, and triangulate-then-join steps; output positions, face rings, corner UVs, and vertex-group weights match Blender for the oracle set. It carries native POINT/CORNER ColorStore layers through their FLOAT_COLOR/BYTE_COLOR source metadata and the existing byte codec, and preserves loop identity for the shared custom-normal code transfer. Generic POINT/EDGE custom data uses Blender's per-collapse INT rounding, BOOLEAN positive-weight OR, FLOAT/FLOAT_COLOR interpolation, and BYTE_COLOR byte conversion/rounding. Native seam/sharp flags follow Blender's rules: COLLAPSE ORs the flags of merged edges, and DISSOLVE keeps the surviving edge's flags as `BM_edge_splice` does.
- Collection instances join their members (and nested instances) into one mesh with `geom/concat.rs`, so each member keeps its own attributes as Blender's per-object meshes do: UV layers (joined by position: a member's k-th layer joins the instance's k-th layer, first-seen name wins, and a member without that layer contributes default values and no coverage), `seam` and other named edge attributes, vertex groups, shading (each member's smooth flags are canonicalized before the join), and the material slot table. Native color layers union by name and preserve order/designation; matching POINT/CORNER layers promote to CORNER with per-corner expansion, missing member values are transparent black, and a FLOAT_COLOR source promotes the joined encoding. Mirrored members reverse winding and per-corner values; out-of-range face material indices clamp to the member's last slot; differing Blender attribute data types promote to the higher-precision type. Positional `blender_*` data of a member that lacks it takes the default. A member's `blender_metadata` custom properties join into the instance mesh, except `potter.*` bookkeeping keys; a conflict fails evaluation of that instance with `UNSUPPORTED_FEATURE` and its `node_id`.
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
| Modifier / graph | `modifier.*`: create / update / reorder / apply all types. `modifier.create` and `modifier.update` take `enabled` (global switch), `show_viewport`, and `show_render`. `params.use_pin_to_last` moves the modifier as Blender's pin setter does: a new modifier is inserted before the trailing pinned block, and pinning or unpinning moves it at once. `modifier.reorder` fails with `INVALID_OPERATION` when Blender's pin or original-data rules would stop the move short of `index`. `graph.*`: all nodes, sockets, links, groups, interfaces, zones, attributes |
| sculpt / paint | `sculpt.*`, `paint.*`: brush / stroke, mask, face set, symmetry, Dyntopo, Multires, texture / vertex / weight |
| UV / material | `uv.*`: unwrap, pin, pack, transform, layer_create, layer_delete, layer_rename, layer_set_active, layer_set_render. `material.*`: all shaders and slots. `image.*`: pixels, UDIM, color space |
| rig / animation | `bone.*`, `pose.*`, `constraint.*`, `driver.*`, `shape_key.*`, `action.*`, `keyframe.*`, `fcurve.*`, `nla.*` |
| physics | `physics.*`, `simulation.*`: system, solver, collision, force, seed, initial state, cache conditions |
| Grease Pencil | `grease_pencil.*`: layer, stroke, fill, frame, attribute, effect, animation |
| render / composite | `camera.*`, `light.*`, `world.*`, `render.*`, `color.*`, `compositor.*`: all settings, graphs, passes, outputs |
| tracking / video | `tracking.*`, `mask.*`, `sequencer.*`: track, solve, spline, strip, retiming, effect, sound |
| asset / library | `resource.*`, `library.*`, `asset.*`: retrieval, reference updates, pack / unpack, link / append, override, metadata |
| script / extension | `text.*`, `property.*`, `extension.*`: source, custom data, dependency registration, permitted native adapters |
| avatar | `avatar.update`, `avatar.clear`: typed `SceneDoc.avatar` settings |

### Avatar Settings

- `SceneDoc.avatar` is optional and includes `armature`, `humanoid`, `meta`, `expressions`, `first_person`, `look_at`, `spring_bone`, and per-material `mtoon`.
- `avatar.update` sparsely replaces top-level components; its first call MUST set `armature`, `humanoid`, and `meta`. Set optional components to `null` to clear them. `avatar.clear` removes avatar settings.
- `humanoid` maps official VRM 1.0 role names to bone IDs, requires 15 bones, and checks ancestor order. `meta` includes name, authors, license_url, restrictive-default VRM 1.0 permissions, and optional `legacy` VRM 0.x overrides.
- Expressions support morph, material-color, and texture-transform binds and overrides. Spring bones support sphere/capsule colliders, collider groups, per-joint settings, and an optional center; scene references use `{node, bone?}`.
- Invalid settings are rejected atomically with a JSON pointer; the revision remains unchanged.

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
- Generate simulation / geometry / texture / animation cache keys from Snapshot inputs, profile, solver, seed, frame, and execution policy. The shared simulation key and `evaluation_hash` include a deterministic potter-core identity of sorted core source files, optional `native/` and `vendor/`, package `build.rs` / `Cargo.toml` / `Cargo.lock`, workspace `Cargo.toml` / `Cargo.lock` when present, and `rustc -vV`, target, enabled features, profile, optimization level, debug mode, encoded rustflags, and target CPU features. Missing build metadata is hashed as an empty value. Changes to these inputs conservatively invalidate the affected cache; engine/solver labels remain semantic identifiers, and preview keys inherit the identity through `evaluation_hash`.
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
- Smooth shading: faces with `shade_smooth`, sharp edges, or authored `custom_normal` codes use Blender 5.2 corner normals (`geom::normals::corner_normals`, evaluated positions, world normal matrix = inverse transpose). Auto/code0 entries follow their current fan; authored codes decode in that fan's current space, and `"explicit_zero"` derives the literal zero vector. Solid, beauty (path and realtime), normal, and AO interpolate the derived shading normals per pixel with barycentric weights; beauty evaluates named Color Attribute / Vertex Color inputs against the corresponding native POINT or CORNER color layer. True displacement moves samples along the shading normals while sub-triangles shade with their own displaced face normal. Flat faces keep their geometric normal. Lighting and normal / AO passes use the shading normal; ray origins, backface decisions, and AO origin side use the geometric normal, and smooth shadow rays start from Hanika's shadow-terminator offset, as in Cycles. Solid has no ambient occlusion, matching Blender Workbench. If corner normals cannot be built for a mesh, it renders flat and reports a `shading.corner_normals_failed` warning in preview and render `warnings`.
- Treat scalar density in VDB / inline volumes as an extinction coefficient [m^-1]. density=1 means 1/m. Calculate transmittance with the Beer-Lambert equation, `T = exp(-density × distance[m])`. Convert ray distance using `unit.scale_length` (meters per scene unit), and sample after applying the Object world transform and VDB grid transform.
- Specify AA, samples, seed, transparent background, overlay, engine / device, and OCIO settings in the manifest. Distinguish sampling policies for color and selection passes.
- The default preview context is the viewport selection context. render uses render visibility and the View Layer. Evaluate exclusions, holdouts, and instances.
- Modifier visibility follows Blender's depsgraph modes. Preview (`preview`, `inspect`, and `apply`, including its previews) evaluates with `enabled` and `show_viewport`; saved renders, `bake`, PLY, STL, USD, and Alembic evaluate with `enabled` and `show_render`. OBJ, glTF/GLB, VRM, and FBX use the viewport stack, matching each Blender writer's default when modifiers are applied. BLEND preserves the stack and both flags; BVH does not evaluate modifiers. SVG/PDF have no Blender writer and use the render stack. `enabled` is the global switch for both. `apply` applies only the target modifier to base data and removes only that modifier; every other modifier remains in order with its flags unchanged, matching Blender. `inspect` and `apply` source data keep every flag; only evaluation chooses the stack. Render manifests record their `visibility` (`render` or `viewport`), and `pick` evaluates the purpose the manifest records.
- `modifier.apply` resolves transform-only object references from world transforms without evaluating their modifier stacks. True geometry operands use their own Preview-active modifier stacks. Array `FIT_CURVE` measures the referenced Curve Data-Block path under its world transform.
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
- `evaluation_hash` includes the deterministic potter-core build identity, so preview keys and reuse change when included source, compiler, target, or enabled features change; engine labels remain semantic identifiers.
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
| vrm | REQUIRED | REQUIRED | Humanoid avatar, skin, morph, MToon, spring bones, metadata |
| vrm0 | REQUIRED | REQUIRED | Humanoid avatar, skin, morph, MToon, spring bones, metadata |
| usd / usda / usdc / usdz | REQUIRED | REQUIRED | Scene, instances, materials, geometry, animation, volume, assets |
| abc | REQUIRED | REQUIRED | Alembic geometry / transform / attribute time-series cache |
| fbx | REQUIRED | REQUIRED | Hierarchy, mesh, material, rig, animation. Report variants / losses |
| obj / ply / stl | REQUIRED | REQUIRED | Geometry, normal / UV / color / material as supported by the format |
| bvh | REQUIRED | REQUIRED | Skeleton, motion |
| svg | REQUIRED | REQUIRED | Curve / Grease Pencil. Specify projection, frame, sampling |
| pdf | — | REQUIRED | Grease Pencil projection, frame, page |
| PNG / JPEG / TIFF / EXR / video / audio | Asset import | Render / bake | Color, bit depth, alpha, codec, frame / fps / audio sync |

- VRM 1.0 uses `VRMC_vrm`, `VRMC_springBone`, and `VRMC_materials_mtoon` 1.0; VRM 0.x uses the `VRM` extension.
- VRM 1.0 and 0.x import and export. `pot validate --format vrm|vrm0 [--strict]` runs the same export preflight without writing a file and reports `VRM_EXPORT_PREFLIGHT_FAILED` when output is unrepresentable.
- VRM import (`pot import --format vrm` for VRM 1.0, `--format vrm0` for VRM 0.x) reads a `.vrm` glTF container with the avatar basis of that version (VRM 0.x adds a Ry(180) turn) and undoes the baked VRM 1.0 mesh scale. The file MUST carry the matching extension (`VRMC_vrm` or `VRM`); the other version is rejected.
- VRM import restores exactly, so a Potter export re-imports without losses and re-exports to the same glTF JSON (float fields within 1e-5 absolute; the first export's source rig solves IK in f32): avatar meta, humanoid, expressions, first person, look-at, spring bones and collider groups, and MToon materials. Root `extras.potter` carries what VRM cannot express and other readers ignore it: `avatar_meta` (VRM 0.x only: name, license URL and copyright; VRM 1.0 decodes its own meta) and `vrm0SpringBone` (VRM 0.x collider-group names and order and each chain's joints).
- VRM import reports a loss, refused without `--allow-lossy`, for each glTF extension Potter neither decodes nor supports (`vrm.extension.<name>`, such as `VRMC_node_constraint`) and for each VRM feature Potter cannot hold: `vrm.meta.*` (including `vrm.meta.thumbnail`), `vrm.humanoid.*`, `vrm.expression.*`, `vrm.first_person.*`, `vrm.look_at`, `vrm.spring.*`, and `vrm.mtoon.*`. Malformed VRM data is a loss, never a silent default.
- Avatar degradation is per component. Meta, armature and humanoid are required: when one is not representable the avatar is dropped and the loss names it (`vrm.meta`, `vrm.armature`, `vrm.humanoid`). Optional components are dropped alone with their own loss and the rest of the avatar is kept: first person (`vrm.first_person`), look-at (`vrm.look_at`), each expression (`vrm.expression.<name>`), each spring chain, collider and collider group (`vrm.spring.*`), and each MToon material (`vrm.mtoon.<material>`). An out-of-range MToon color (outside 0 to 1) is clamped and reported as `vrm.mtoon.<field>`.
- glTF and VRM texture graphs: a material whose texture has a `KHR_texture_transform` (any glTF slot), or that has an emissive or occlusion texture, or a `normalTexture.scale` other than 1, becomes a material shader graph, as Blender's glTF importer builds it. A transformed texture is an Image Texture whose `Vector` comes from a Mapping node (POINT) fed by Texture Coordinate `UV`; Mapping Location is `(ox + sy sin r, 1 - oy - sy cos r)`, Rotation z is `r` and Scale is `(sx, sy, 1)`. The base color and emission textures multiply by their factors on the Principled BSDF; a grey emissive factor folds into Emission Strength, as Blender does. Metallic and roughness multiply by their factors. Occlusion passes through Separate Color (Red) into the `glTF Material Output` group, which the exporter reads back as `occlusion`. `normalTexture.scale` is the Normal Map Strength. The texture fields are cleared, since the graph holds the textures. A non-default `occlusionTexture.strength` is reported as `material.occlusion_strength`: Potter applies occlusion at full strength.
- glTF packed metallic-roughness: a `metallicRoughnessTexture` always becomes a material shader graph, as Blender's glTF importer builds it. Separate Color Blue feeds Metallic and Green feeds Roughness, each multiplied by its factor. Potter's native metallic and roughness fields read only the Red channel, so they cannot hold a packed texture; the texture fields are cleared, as for the other graph textures.
- glTF image colourspace follows Blender's importer (`io_scene_gltf2` `blender/imp/texture.py`, `is_data`). An image used as data, meaning the metallic-roughness, normal, or occlusion texture of a material that a primitive references, imports `NonColor`, so its values are read raw. Every other image imports `Srgb`. Blender keeps one image per glTF image and nothing resets `is_data`, so an image that is both a colour and a data texture is `NonColor` everywhere. Materials that no primitive references contribute no data uses, as Blender creates them only on demand. The glTF sampler's default wrap is REPEAT, which Potter's graph path follows; the native material fields clamp at the image edge, which differs from Blender at the texture border.
- glTF texture UV sets: each `TEXCOORD_n` imports as the layer `UVMap` (n = 0) or `UVMap.00n`, up to the 8-layer limit. A texture reads the layer its `texCoord` names, or the layer named by a `KHR_texture_transform` `texCoord` override on that texture; set 0 is the render layer (`null`). Sets past the limit import as the render layer with a `mesh.uv_layer` loss. VRM extension data continues to apply its own per-component degradation.
- Appending a glTF import remaps each material's shader graph and each Image Texture's image reference through the scene ID maps, so appended graph materials keep their own images. VRM expression color binds validate after MToon materials are accepted. The native evaluator rejects Separate Color modes other than RGB with `UnsupportedFeature`.
- VRM 0.x MToon main texture: `_MainTex` (offset x, stored offset y, scale x, scale y), which the VRM Add-on reads (`vrm0_importer.py`, v4.10.2, the version pinned in `tests/vrm-consumer/blender/vrm-addon.lock.json`), or Potter's `_MainTex_ST` (scale x, scale y, offset x, stored offset y) when `_MainTex` is absent. Its stored offset y becomes `1 - offset y - scale y`. As the add-on does, it replaces the base color texture's own `KHR_texture_transform`, and a scale whose magnitude is below `f64::EPSILON` is 1. The glTF import applies it, so it holds whether or not the material needs a graph for another texture, such as a packed metallic-roughness. A replaced non-identity `KHR_texture_transform` is reported as `vrm.mtoon._MainTex`. A transform on a material with no base texture is reported as `vrm.mtoon._MainTex`, or as `vrm.mtoon.main_tex_transform` when it came from `_MainTex_ST`.
- VRM 1.0 MToon texture slots carry their `KHR_texture_transform` (`avatar.mtoon.<material>.texture_transforms`, keyed by slot field name). VRM 0.x has no per-texture transform, so VRM 0.x export refuses a non-identity one (`vrm.v0.mtoon_texture_transform`).
- VRM 0.x texture-named vectors (`_MainTex`, `_UvAnimMaskTexture`, and other texture properties) that are `[0, 0, 1, 1]` are untransformed, as the VRM Add-on writes them, and import without a loss.
- VRM 0.x spring bone groups list root bones. Each root not nested under another listed root starts one chain that follows its glTF subtree. Writers that list every bone of one chain (the VRM Add-on) therefore import as one chain, and independent roots (VRoid hair) import as one chain each.
- Known VRM import limits, not restored exactly: a VRM 0.x MToon texture the exporter channel-remapped to gray imports as that gray copy (named as its source image) unless the original image is also in the file; a VRM 0.x mesh shared by several nodes is ambiguous for annotations and morph binds; an empty VRM 0.x first-person annotation list cannot be told apart from an absent one.
- glTF / GLB / VRM node IDs: a node takes `extras.potter.id` when present, else its glTF name, else `node`. The ID is lowercased, characters outside `[a-z0-9_-]` become `_`, a result that does not start with a letter gets a `node<index>_` prefix, and the result is truncated to 64 characters. A duplicate gets `_<n>`, where `n` is the node's index in the glTF `nodes` array (increasing until unused). The first node in file order keeps the plain ID, so the same file always imports with the same IDs.
- glTF / GLB / VRM skins share armatures as Blender's importer does. A skin's armature is the deepest common ancestor of its joints and skeleton, or that node's parent when the ancestor is itself a joint (`None` for the scene root); skins with the same armature form one armature Data-Block whose bones are the union of their joints, parented to the nearest ancestor joint in that union. A Potter export names its armature node in `extras.potter.armature_id` and uses it instead. Separate characters under a shared empty root therefore import as separate armatures.
- A glTF node parented to a skin joint imports as a bone-parented Potter node. Its converted glTF world transform is expressed relative to the imported bone's rest-world matrix, preserving accessory placement across VRM 1.0 and 0.x coordinate frames.
- Potter vertex metadata (`extras.potter.vertex_ids`, and the shape-key, vertex-weight and polygon records keyed to it) describes the vertices of one Potter export. When the IDs no longer match the mesh vertices, another tool has edited the file (the VRM Add-on keeps mesh extras and drops root ones). Such a mesh imports as third-party glTF with a `mesh.vertex_ids` loss, and its stale polygon record reports `mesh.polygon_topology`.
- Morph NORMAL deltas map as the difference of the mapped morphed and base normals, so a non-uniform baked scale round-trips exactly.
- VRM import tolerates f32 rounding in frames: a spring radius frame must be a uniform scale (equal axis lengths, orthogonal axes) within `F32_FRAME_TOLERANCE` (1e-4 relative), and the exporter's exact-f64 check uses 1e-8 with the same function.
- Reject settings a selected VRM version cannot represent; never silently drop them. VRM 0.x supports only sphere colliders (`vrm.v0.spring_capsule` for capsules), one set of joint settings per spring, and no expression overrides; combined MToon scroll and rotation and some MToon textures are also unrepresentable.
- Never broaden licensing rights. A VRM 0.x Creative Commons `licenseName` is rejected when canonical `modification` is `prohibited`, when `allow_modification` (private only) meets a license that permits sharing adapted material (CC0, BY, BY-SA, BY-NC, BY-NC-SA), or when a license that permits commercial use by anyone (CC0, BY, BY-SA, BY-ND) meets a `commercial_usage` other than `corporation` (`vrm.v0.meta_license_modification`, `vrm.v0.meta_license_commercial`).
- Reject unsupported shared mesh constructs: `mesh.shape_key_absolute`, `mesh.skin_unweighted_vertex` (a skinned vertex with no deform-bone weight, since glTF requires weights summing to 1), and `vrm.skin_influence_limit` (VRM only: more than four influences on one vertex). A texture that names a missing UV layer is not rejected: it falls back to the render layer and is reported as `uv.layer_missing`.
- Shading refusals (`UnsupportedFeature`, typed `feature_id`): `modifier.bevel.mark_seam` (Bevel `mark_seam` on a mesh with seam edges), `modifier.bevel.shading_segments` (Bevel `harden_normals`, or `mark_sharp` on a mesh with sharp edges, with more than one segment or `affect` VERTICES), `modifier.bevel.high_valence_vertex_mesh`, and `modifier.boolean.downstream_topology`. Each stays silent where Blender does nothing: an unmirrored bisect axis, `mark_seam` without seams, `mark_sharp` without sharp edges. Triangulate applies `min_vertices` as Blender does and refuses nothing for it.
- Mirror (`geom/modifiers/mirror.rs`) follows Blender 5.2.2 per axis, as the CHANGELOG describes: a bisect on a mirrored axis is evaluated, not refused. `mirror_object` copies through the owner-to-mirror-object transform built from the scene's world matrices, as `mesh_mirror.cc` builds it. Edge attributes follow Blender's CustomData rules: a cut edge's two halves keep the source edge's attributes (`BM_edge_split`), a bisect chord has no source edge and takes BMesh defaults (`BM_face_split` with a null example edge), and a mirrored copy takes its source edge's attributes (`mesh_mirror.cc`). During merge, the weld context deduplicates affected faces by mapped edge sets and keeps the earliest face's material/corner provenance; untouched and merge-off faces are not deduplicated. `use_clip` has no evaluated effect, and copy UVs flip on the mirrored axes.
- Remesh follows Blender 5.2.2: BLOCKS, SMOOTH, and SHARP use sparse-octree DualCon scan conversion, open-ring patching, and in-octree disconnected-component flood fill; open planes and some non-manifold sheets evaluate to empty geometry, while open boxes and cylinders are patched into closed output. VOXEL supports OpenVDB 13 adaptive meshing (`adaptivity` 0 through 1), including open and non-manifold inputs. Empty results are valid evaluated meshes and remain legal through inspect, apply, preview/render, persistence, and supported exports. DualCon uses a combined 4,000,000 input/scratch/output work-item budget; budget excess returns `LIMIT_EXCEEDED` rather than allowing unbounded arena growth.
- Modifier property names, types, enum values, and defaults are defined by the Blender 5.2.2 runtime table in `param_table.json`; defaults are probed from the owning object type (Volume for Volume-only modifiers). Operations MUST validate schema-defined properties rather than a separate op-layer property whitelist. Evaluators use schema defaults when a property is omitted; a typed unsupported refusal is reserved for an evaluator-specific behavior that cannot be performed, not a table annotation applied to every non-default value.
- Behavioral regression coverage is deliberately scenario-based, not an assertion that arbitrary non-default values must alter a box. `modifier_param_audit.rs` compares Blender 5.2.2 and Potter for activated `edge_split.use_edge_angle=false` and `triangulate.min_vertices=5`, `solidify.solidify_mode=NON_MANIFOLD` (now equal to Blender), the measured `decimate.ratio=0.5` gap, the default screw angle, Blender-confirmed no-ops for `array.fit_length` under `FIXED_COUNT` and `array.use_apply_on_spline` on a mesh, and Cast (schema defaults, `CYLINDER`, `CUBOID` with and without an explicit size, radius and size paths). Cast's evaluator follows Blender's `MOD_cast.cc`: radius 0 is unbounded, a zero size is the mean vertex distance (10 when every vertex is at the origin), and `use_radius_as_size` selects the radius; negative values are refused. The uneven-mesh and all-zero Cast oracle lives in `modifiers_more.rs`. The runtime-default test checks every recorded default against Blender; it does not evaluate them. `diagnostic_inventory.json` names unexercised parameter families and owners; it is not a test allow-list, and an unexercised parameter is not claimed as a no-op or a parity gap.
- Full Solidify parity is covered by the 47-case Blender-generated corpus in `tests/fixtures/solidify/` (EXTRUDE and NON_MANIFOLD), including unrelated deform weights with empty or missing thickness-group names, a null material-slot layout, bevel-weight presence and values, the zero-area collinear triangle, and sharp/seam provenance on two-group connectors and larger-group closing edges. The in-crate parity harness is `geom/modifiers/solidify/parity_tests.rs`.
- Construction modifier behavior is checked against independent Blender 5.2.2 fixtures in `tests/fixtures/construction_p/`: Array copy transforms, fit boundaries, caps, merging and UVs; Build frame boundaries and seeded face/edge/point order; Screw object-axis and offset, iteration, merge, winding, shading and UV behavior; Weld group selection and connected/all merging; Wireframe group-weighted offsets and object-slot material clamps. Tests compare geometry and preserved mesh attributes, not parameter wiring.
- glTF / GLB / VRM meshes write `NORMAL` from Blender 5.2 corner normals: `shade_smooth` faces, `sharp` edges, and `custom_normal` corners (all-smooth meshes use vertex normals; otherwise fans split at sharp, boundary, non-manifold, and flipped edges). Morph targets carry `NORMAL` deltas; a corner whose morphed normal is undefined keeps its base normal.
- glTF / GLB node IDs come from node names where no `extras.potter.id` exists; see the node ID rule under VRM import. Plain glTF files that previously imported as `node`, `node_1`, ... now import with their names.
- Non-glTF exchange formats carry Blender 5.2 corner normals (`shade_smooth`, `sharp`, `custom_normal`) in the layout Blender 5.2 writes by default; import derives `shade_smooth` (a face with any corner differing from its flat normal) and `custom_normal` (corners differing from the computed shading), so sharp edges round-trip through their corner normals:
  - OBJ: world space with Blender's default axes (file `(x, z, -y)` for Potter `(x, y, z)`). Writes deduplicated `vn` and `f v//vn`, with `s 0` / `s 1` marking each face's smooth state. Partial `vn` reports `mesh.normals`; smooth groups without `vn` report `mesh.smoothing`.
  - USD / USDA / USDC / USDZ: object-local `normal3f[] normals` with `interpolation = "faceVarying"`, and `uniform token subdivisionScheme = "none"`. Import accepts faceVarying, vertex, varying, uniform, and constant interpolation; other values report `mesh.normals`.
  - FBX (ASCII and binary): object-local `LayerElementNormal` (`ByPolygonVertex`, `IndexToDirect`), with no `LayerElementSmoothing`. Import reads `ByPolygonVertex` and `ByVertice` mappings; smoothing without normals reports `format.fbx.smoothing`.
  - Alembic: object-local `N` (faceVarying) from the same corner normals, in Blender's Y-up basis: points, normals, and local transforms are stored as `(x, z, -y)`, and each polygon's winding is reversed, as Blender writes it. Shading attributes export without `--allow-lossy`.
  - PLY: world space, one normal per vertex, written for every mesh in the file once any mesh is shaded. The value is Blender's vertex normal: angle-weighted face normals, or the normalized sum of corner normals when custom normals exist. A file with no shaded mesh writes no normal properties, which matches Blender's default export (`export_normals` off). A vertex whose corners disagree in a shaded file is lossy (`mesh.shading`).
  - STL: facet normals are the winding normal. A face whose corner normals differ from its flat normal is lossy (`mesh.shading`). Import ignores stored facet normals as Blender does; a stored normal that contradicts the winding reports `mesh.normals`.
  - World-space formats (OBJ, PLY, STL) reverse the winding of mirrored nodes so faces agree with normals. Object-local formats keep winding and carry the transform, as Blender does.
  - Zero-scale nodes and degenerate faces export without error. Zero-area faces and fans whose normals cancel take the normal Blender computes for them; OBJ writes zero `vn` for a singular object, PLY uses the cofactor normal matrix.
- Write one primitive per material slot used by `Face.material_index`; primitives share vertex and morph accessors. A collection-instance mesh indexes one combined slot table (`Snapshot.slot_bindings`), including each member's material, source render UV index, UV availability and joined color metadata; nested bindings remap into the outer name union. Two members sharing a material each get a primitive, as Blender's per-object meshes do. A slot without material writes no `material`. Default selectors use each source member's render layer; an explicitly named layer missing on that member falls back to the member's render layer with `uv.layer_missing`. When the same material resolves to different UV indices, export writes a material variant so each primitive references its member's `TEXCOORD_n`. When VRM 1.0 bakes a negative scale into geometry, triangle winding is reversed so front faces stay front; the stored polygons are reversed the same way.
- Material export (glTF / GLB / VRM 1.0 / VRM 0.x) resolves every material to one glTF surface (`exchange/material_surface.rs`) before any writer runs; the writers read only that surface. A material without a custom node graph exports its explicit fields. Reroute nodes are stepped over, as Blender's exporter does; muted nodes and nested node groups are not (Blender does not step over muted nodes either; nested groups other than `glTF Material Output` are unsupported_node losses). A custom graph (imported from `.blend` or authored with node ops) is authoritative: the material's mirror fields and native texture references are ignored, and native texture references on such a material report `material.shader_graph.material_textures`. For a graph material, `alpha_mode` is Blender's render method, which Potter's renderer reads; it is not an export field. glTF `alphaMode` comes from the graph's Alpha path (Principled Alpha, `Mix(Fac, Transparent, X)`, clip patterns), as Blender's exporter does. Conversion follows Blender 5.2's glTF exporter: Principled BSDF inputs, Image Texture with UV Map and Mapping (`KHR_texture_transform`, using Blender's conversion), Normal Map (tangent space, Strength as `normalTexture.scale`), Separate Color and multiply Mix packing into `metallicRoughness` and `occlusion`, Emission and Add Shader, Background as `KHR_materials_unlit`, `Mix(Fac, Transparent, X)` and clip patterns (Math Greater/Less Than, Round, Subtract) as alpha.
- Materials that no node references never block export: an invalid graph on an unused material exports its explicit fields.
- Documented divergences from Blender's exporter, where glTF can express the correct value: alpha from `Mix(Fac, Transparent, X)` is `Fac` (Blender drops it); Background strength folds into the unlit color when every product component is at most 1 (Blender drops strength); sheen color is the tint times the sheen weight (Blender omits the weight); `Mix(Principled, Emission)` keeps the Principled base and weights the emission by the mix factor (Blender adds the emission at full weight).
- Losses for graph parts glTF cannot express (`--allow-lossy` required; the export uses the closest value): `material.shader_graph.unsupported_node` (node types outside the supported set, e.g. Noise and Color Ramp), `material.shader_graph.unsupported_input` (Subsurface, Thin Film, Coat IOR or Tint, Diffuse Roughness, linked IOR, textured anisotropy), `material.shader_graph.unrepresentable_surface` (two surfaces, a non-Principled BSDF surface, Background mixed with Principled), `material.shader_graph.normal_map` (Bump, OBJECT or WORLD normal maps, a texture wired directly into Normal), `material.shader_graph.transform` (Mapping rotation about X or Y), `material.shader_graph.vertex_color` (vertex-color use outside representable base-color multiplication), `material.shader_graph.material_textures` (native TextureRef fields alongside a custom graph), `material.shader_graph.displacement`, `material.shader_graph.volume`, and `material.shader_graph.sampler` (Cubic or Smart interpolation, CLIP extension). An invalid graph (dangling link, missing node or image, cycle, no OutputMaterial) is an error.
- Textures: each glTF channel (R, G, B, A) is fed from one channel of one image or is the constant 1.0. A texture slot samples only the channels glTF reads for it (base color RGB, plus A when the material is not opaque; metallicRoughness G and B; occlusion R; normal and emissive RGB; and so on). When every sampled channel comes from one image at its own channel, the image is reused as is, with the unsampled channels taken from that image (glTF never reads them). This is Blender's happy-path rule (`encode_image.py`); Blender instead fills unlinked channels with white, so a sampled constant forces a derived PNG here too, as in Blender. Otherwise one derived PNG holds the raw, encoded values. Samplers follow Blender: Linear uses `magFilter` 9729 and `minFilter` 9987; Closest uses 9728 and 9984; wrap maps to REPEAT, CLAMP_TO_EDGE, or MIRRORED_REPEAT.
- `alphaMode` is written for every glTF material: `BLEND`, or `MASK` with `alphaCutoff` (`OPAQUE` is the default and omitted).
- Emission: `emissiveFactor` is the emission color times the strength. `KHR_materials_emissive_strength` is written only when a channel exceeds 1 (the peak is the strength). For materials without a node graph, `extras.potter` keeps the native `emission_strength`, `transmission` and `ior`; graph materials omit them, because their graph decides those values and the importer reads the glTF extensions.
- `KHR_materials_ior` is written only with `KHR_materials_transmission` or `KHR_materials_specular`, and only when the value is not 1.5. `KHR_texture_transform` is listed in `extensionsRequired` whenever a texture uses it, as Blender does. `KHR_materials_specular` never writes a factor above 1: Blender's doubled Specular IOR Level moves its excess into `specularColorFactor`, and a textured level reads the raw texture alpha as Blender does.
- VRM 1.0 MToon takes the base color, emission, normal, alpha, and double-sided state from the glTF surface; MToon rejects unlit, clearcoat, sheen, specular, anisotropy, and transmission surfaces (`vrm.mtoon_<name>`). VRM 0.x writes `_MainTex_ST` from the base texture transform (V is flipped, as UniVRM imports it), `_BumpScale` from the normal scale, `_EmissionMap`, `_Cutoff`, and blend and cull state from the same surface; a rotated base transform is `vrm.v0.mtoon_texture_transform`.
- `.blend` export turns native material texture fields (`base_color_texture`, `roughness_texture`, `metallic_texture`, `normal_texture`) into safe Image Texture nodes linked through UV Map nodes to the named `TextureRef.uv_layer`, or the render layer when null. Base color and its alpha factor feed Principled Base Color/Alpha; roughness and metallic use Non-Color data; normal textures use a Normal Map node. Images resolve through the document's image references so packed/external resources remain linked after reopen. Native material scalar fields remain on the Principled BSDF; unsupported features report a loss, and embedded scripts are never executed during conversion.
- Native material texture fields sample as the Image Texture node they stand for, through the same sampler as an Image Texture node: the image's colourspace decodes its values (Srgb linearised, Linear and NonColor read as stored), `TextureRef.interpolation` selects Linear or Closest, and `TextureRef.extension` (`repeat`, `extend`, `clip`, `mirror`; default `repeat`, omitted from scene JSON when default) selects the edge rule as Blender's Image Texture `extension` does. A glTF sampler's wrap maps to the extension as Blender's importer does: CLAMP_TO_EDGE is `extend`, MIRRORED_REPEAT is `mirror`, and an absent wrap is `repeat`. Unknown glTF wrap values clamp as Blender's mixed-wrap importer does and report `material.shader_graph.sampler`; a `clip` field likewise reports that loss when exported because glTF has no CLIP. glTF exports the extension back as Blender's exporter does (`exp/sampler.py`): `extend` and `clip` are CLAMP_TO_EDGE, `mirror` is MIRRORED_REPEAT, and `repeat` writes no wrap.
- glTF textures whose S and T axes wrap differently cannot be a field or one node extension. They become a material graph, as Blender's importer builds them: UV splits into X and Y (Separate XYZ), each axis is WRAP for REPEAT or PINGPONG for MIRRORED_REPEAT (a CLAMP axis passes through), the axes recombine (Combine XYZ) into the Image Texture's Vector, and the node's extension is EXTEND. The glTF exporter recognises this per-axis pattern as Blender's exporter does (`exp/sampler.py` `detect_manual_uv_wrapping`): a pass-through axis takes the node's extension, a WRAP with max 0 and min 1 is REPEAT, and a PINGPONG of scale 1 is MIRRORED_REPEAT. The sampler written back is the one imported, so a mixed-wrap round trip is exact. Potter's sampling of these graphs matches Blender 5.2.2's render at every pixel centre of a UV ramp from -0.5 to 1.5 (`tests/fixtures/texture_wrap`); both sides interpolate raw values, so the check isolates wrap semantics from the sRGB decode order (see the sRGB sampling rule below).
- An sRGB-coloured image keeps its encoded texels for every lookup from Image Texture nodes, native material fields, the world environment, and the vertex-weight mask. Linear sampling interpolates those encoded values and decodes the result, and closest sampling decodes the single texel, as Cycles does (`svm_image_texture`, `NODE_IMAGE_COMPRESS_AS_SRGB`); the decode is Cycles' f32 `color_srgb_to_linear_sse2` (`fastpow24_sse2`). Non-sRGB images are read as stored. The vertex-weight mask follows Blender's `BKE_texture_get_value` order (`MOD_weightvg_util.cc`): the colour channels are decoded after mixing, and the `INT` (Intensity) channel is the mean of the encoded RGB with no decode. Generated and painted sRGB images are read as encoded on every source: `.blend` export writes their float values encoded, and glTF export writes 8-bit sRGB PNG, which Blender decodes after quantising to bytes. `tests/fixtures/srgb_texture` checks the Image Texture path against Blender 5.2.2 renders at Blender's sample UV, with tolerance 2e-6. EEVEE and the viewport decode 8-bit sRGB textures in hardware before filtering, so they differ from the Cycles order that `pot preview` and renders use.
- Image Texture colour reads and material texture fields follow Cycles' alpha rule. A straight-alpha image is premultiplied in its stored buffer, in the arithmetic of its precision (integer `floor(c * a / max)` for 8-bit and 16-bit, a half product rounded to half for half, a plain product for float; `util_image_multiply_native`, `intern/cycles/util/image.h`), so a colour read whose Alpha output is unlinked returns the premultiplied colour, and a linked Alpha output unassociates it before decoding (`intern/cycles/kernel/svm/image.h`, `NODE_IMAGE_ALPHA_UNASSOCIATE`). Material texture fields take the link state the `.blend` exporter writes: the base colour's Alpha is linked for a non-opaque material, and roughness, metallic, and normal are not. Non-colour images are read as stored. Modifier masks read straight values. Verified by the Blender 5.2.2 oracle set in `tests/fixtures/alpha_texture`.
- EXR images are associated (premultiplied). Blender reads an EXR without `oiio:UnassociatedAlpha` as associated alpha (`image_metadata.cpp`), so `image.load` gives an EXR the premultiplied alpha mode, and its values are read as stored, for single-layer and multilayer files alike. Every Potter EXR writer stores associated alpha, colour times alpha: the render-frame EXR output, the compositor's single-layer and multilayer outputs, and the `.blend` pack, which passes the stored values with the matching datablock alpha mode. Blender's OpenEXR writer copies the float buffer as given (`source/blender/imbuf/intern/openexr/openexr_api.cpp`, `imb_save_openexr`), and a render's Combined pass is premultiplied. `ImageData::straight_pixels` is the one explicit conversion for readers that need straight colour.
- Alpha mode None is Blender's `IMA_ALPHA_IGNORE`. The loader sets alpha to 1 for an RGBA image (`image_metadata.cpp`, `ignore_alpha`), no premultiply happens, and no colour is unassociated (`intern/cycles/scene/shader_nodes.cpp` leaves IGNORE out of the unassociate flag). Every read of such an image returns alpha 1, and the stored colour is returned as stored.
- Image input precision: 8-bit and 16-bit PNG, 8-bit and 16-bit TIFF, half EXR, and float EXR load with the precision of their file, through the image crate's decoder colour type for TIFF and the other image formats (`decoded_precision`) and the EXR channel types for EXR (`exr_precision`). Each of these is oracled in `tests/fixtures/alpha_texture`, including 16-bit TIFF (Blender loads it as a ushort buffer, and its premultiply is the integer floor). A multilayer EXR written by Blender from a render is also oracled, through its Combined layer.
- The `.blend` export writes each Image Texture node's `extension` from the texture (EXTEND, CLIP, or MIRROR), and writes no extension for REPEAT, so Blender's default applies.
- FBX follows Blender's `io_scene_fbx`: on import a texture whose `WrapModeU` or `WrapModeV` is non-zero takes EXTEND, and REPEAT otherwise (`import_fbx.py` `texture_mapping_set`); on export `WrapModeU` and `WrapModeV` are written as clamp (1) for any extension other than REPEAT (`export_fbx_bin.py`). Since FBX cannot represent MIRROR or CLIP, either texture extension reports `format.fbx.texture_extension` rather than silently round-tripping as EXTEND. The Math node's WRAP and PINGPONG follow Blender's `wrapf(value, max, min)` and `pingpongf(value, scale)` (`blenlib/intern/math_base_inline.cc`), with the second input as max or scale and the third as min, which is the socket order Cycles uses (`intern/cycles/kernel/svm/math_util.h`). The importer's WRAP (max 0, min 1) therefore wraps into (0, 1] as Blender does.
- Blender-imported meshes store smoothing, sharpness, seams, custom normals, colors, and generic attributes in native mesh data. `shade_smooth`, `sharp`, `seam`, and `custom_normal` are the only live shading representation; custom normals persist Blender code pairs/Auto status and optional loop-reference order, while vectors are derived for rendering/export. Blender's `position`, cached topology, and mirrors (`sharp_face`, `sharp_edge`, `uv_seam`, `material_index`, raw compatibility `custom_normal`) are omitted because Potter has geometry, face material indices, or native counterparts. Selection/edit state remains compatibility data. glTF writes ordered color layers as `COLOR_n`, with the material's selected color layer at `COLOR_0`; unsupported generic attributes are `mesh.attribute` losses. glTF writes `TEXCOORD_k` for each UV layer in order; formats unable to preserve mesh data report format-specific losses, honored by `--allow-lossy`. `.blend` reports `blender.mesh.custom_normal_explicit_zero` for a legacy zero-vector sentinel and `blender.mesh.custom_normal_corner_order` when stored loop order cannot be expressed by Blender polygons. glTF/GLB/VRM report `mesh.normals` when the sentinel maps to their up-vector fallback.
- glTF writes `TEXCOORD_k` for every layer in order (k < 8), and each material's `texCoord` is the index of the layer its texture names (null: the render layer), as Blender's exporter writes it; glTF stores V as Potter does, so there is no flip. OBJ and PLY write the active layer only (`vt`, and `s`/`t` with V flipped as Blender writes it); other layers are `mesh.uv_layer` losses. STL has no UVs, so every layer is a `mesh.uv_layer` loss. USD writes `primvars:st` for the render layer and `primvars:<name>` for the others; FBX writes `LayerElementUV` per layer; Alembic writes one vector GeomParam per layer; `.blend` writes every layer with its active and render index. Each of these flips V as Blender's exporter does for that format.
- glTF import welds vertex copies that share `extras.potter.vertex_ids` back into one Potter vertex, so Potter exports re-import with their source vertex set. Morph `NORMAL` targets are compared with normals derived from the imported topology for every file; disagreement reports `mesh.morph_normals`.
- Potter glTF / GLB / VRM files also carry `extras.potter.polygons` (`version: 1`). `sizes`, `corners`, `primitives`, `smooth`, and the optional `sharp_edges` are each a bufferView index of tightly packed little-endian unsigned ints. `sizes` gives each face's corner count and `corners` its global glTF vertex indices in corner order, faces in `Mesh.faces` order; a mirrored bake reverses each face's corners. `primitives` names the primitive holding each face's triangles, `smooth` is one byte per face, and `sharp_edges` lists each sharp edge once as a glTF vertex pair (absent when there are none). Each field's width is the smallest of 1, 2 or 4 bytes that holds its largest possible value: the vertex count for `sizes`, `corners` and `sharp_edges`, the primitive count for `primitives`, and one byte for `smooth`. The primitives still hold triangles for other readers. Import gives each face the next `corners - 2` triangles of its primitive and restores quads, n-gons, face and corner order, face materials, UVs, `shade_smooth` and `sharp`, adding `custom_normal` only where NORMAL disagrees with the restored shading. Structurally invalid records (bad bufferViews, lengths, out-of-range indices, duplicate corners) fail the import. A well-formed record that does not match its triangles, or an unknown `version`, imports the triangles and reports `mesh.polygon_topology`. Files without `polygons` (third-party, or Potter output from before this record) import triangles as before.
- VRM 0.x `gravityDir` uses UniVRM's frame, VRM 0.x `(x, y, z)` = VRM 1.0 `(-x, y, z)`. Node-local collider offsets and `firstPersonBoneOffset` are written in the VRM 0.x node frame with Z reflected, as UniVRM and three-vrm read them.

- Specify each format's representation scope in the formats catalog. Do not claim preservation of creation features that do not exist in the target format.
- .blend is a round-trip format for the entire creation project. GLB, etc. are interchange formats for assets / geometry / motion, etc. as defined by the formats catalog. Determine the preservation scope for the selected format in advance.
- Record in conversions for that format contract when an interchange format does not include the creation graph, when generating evaluated geometry for primitives / Modifiers, or when triangulating / converting bases / quantizing to float32. The basic box-to-GLB workflow does not require allow-lossy.
- Reject by default conversions that lose targeted geometry, materials, rig, motion, etc. Include feature_id, data_id, reason, and alternatives in losses. Permit only explicitly specified sampling / shader approximation / omission, etc. with `--allow-lossy`. Do not implicitly flatten a .blend creation graph.
- Non-`.blend` export requires evaluation of all targets in the selected context. If an Object cannot be evaluated, fail with `UNSUPPORTED_FEATURE` and return `error.details.feature_id` and `error.details.node_id`.
- .blend export reconstructs all creation data rather than converting the selected frame to mesh output. Preserve hidden / unused data and Action / NLA / physics settings for all frames.
- Evaluated exports follow the Blender writer's default modifier mode: GLB/glTF, OBJ, VRM, and FBX use viewport modifier visibility; PLY, STL, USD, and Alembic use render modifier visibility. Object visibility and required ancestors still apply. Preserve skin / morph / animation / texture as exportable structures.
- Convert potter Z-up world(x,y,z) to glTF(x,z,-y). Convert transforms, normals, bones, bind matrices, and animation in the same basis; retain m. [glTF 2.0 Specification](https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html)
- VRM maps Potter Z-up / -Y-front to +Z-front in 1.0 and -Z-front in 0.x. Normalize bone local rotations; VRM 0.x uses identity rotation and unit scale. Bake nonuniform mesh scale into geometry for VRM 1.0.
- .blend restores Z-up, original Scene units, rotation mode, parent inverse, and shared data. Write materials, Modifiers, nodes, and rigs as creation data.
- .blend export writes each material's texture fields (base color, roughness, metallic, and normal) as Image Texture nodes that read their own UV layer; a normal texture drives a Normal Map node, and each image keeps its colorspace. An authored shader graph is copied, with the fields overlaid on its Principled BSDF, and the source document is not changed. A texture field that cannot be written, because the active output is not driven by a Principled BSDF or a normal texture meets a linked Normal input, is reported as `material.shader_graph.material_textures` (export requires `--allow-lossy`).
- Unpacked .blend export copies each linked image's source file to `<stem>_assets/` beside the output and points the Blender image at that copy; a reopened image loads from that sidecar.
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
