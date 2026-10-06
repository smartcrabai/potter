# Blender Compatibility Requirements

The first release MUST meet these requirements. Apply them together with the [potter specification](spec.md). Every feature in the table is in scope for initial implementation and acceptance verification. Features that can only be saved MUST NOT count as supporting editing and evaluation.

Compatibility baseline: Blender 5.2 LTS. Initial verification build: 5.2.2. Pin the version and build hash in the compatibility profile. Current LTS and release information: [Blender LTS](https://www.blender.org/download/lts/).

"All features" means the standard creation features and bundled features of the baseline version. Replace Blender GUI operations with CLI/JSON operations. Preserve saved workspace, screen, and UI settings across .blend round trips. Detect external extension dependencies by name, version, and feature, and use a registered adapter. Do not infer execution compatibility for future versions or extensions that are not installed.

## 1. Initial Implementation Feature List

The following are potter requirements derived from Blender's official feature categories. Verify editing, evaluation, and output results, not only matching API names. [Blender 5.2 Manual](https://docs.blender.org/manual/en/5.2/index.html)

| Category | Required Features | Information Preserved Through Editing, Evaluation, and Round Trips |
| --- | --- | --- |
| Scene | Multiple Scenes, View Layers, Collections, Collection instances | Active Scene, hierarchy, multiple Collection memberships, exclusions, viewport/render visibility, Scene instances |
| Data management | Object/Data-Block separation, linked duplicates, single user, orphan, fake user | Sharing relationships, reference counts, names, custom properties, asset metadata, typed attributes |
| Editing history and recovery | undo/redo, immutable snapshots, restore after restart, branches | Operation/reference asset/state correspondence, monotonically increasing revisions, atomic commits, invalidation of old caches |
| Transform | All Euler rotation orders, quaternion, axis-angle, negative scale, zero scale, shear, origin | Local/world matrix, parent inverse, delta transform, Object/Bone/Vertex parenting, unit settings |
| Mesh | Vertices, edges, faces, triangle/quad/n-gon, loose geometry | Winding, holes, sharp, seam, crease, split/custom normals, material slots, attribute domains |
| Mesh editing | extrude, inset, bevel, loop cut, bridge, merge/weld, split, dissolve, delete, fill, subdivide, triangulate, knife, bisect, remesh, mirror, symmetrize, proportional edit | Target element IDs, numeric parameters, boundary and connectivity, attribute interpolation. ID correspondence when topology changes |
| Modifier | All standard Modifiers in the baseline version, stack order, enable conditions, apply/copy | Boolean, Mirror, Array, Bevel, Subdivision, Solidify, Shrinkwrap, Decimate, Remesh, Armature, Multiresolution, etc. Compare enum lists and counts |
| Curves and surfaces | Bézier, poly, NURBS, surface, hair curves | Handles, knots, weights, resolution, tilt, radius, bevel/taper, cyclic, attributes |
| Other shapes | metaball, text, lattice, point cloud, volume, empty | Font, text layout, basis/resolution, point attributes, OpenVDB grid, display and instance settings |
| Geometry Nodes | All standard nodes and sockets, node groups, fields, instances, named attributes | Interface, default values, link order, capture, repeat/simulation zones, bake, dependencies, provenance of generated data |
| Sculpting | All standard brushes, masks, face sets, symmetry, Dyntopo, voxel remesh, Multiresolution | Brush assets, sample sequence, pressure/radius/falloff/seed, sculpt layers, base/high-resolution shapes |
| Paint | Texture/vertex/weight paint, clone, stencil, projection | Brush, stroke, paint mask, UV, image pixels, color attributes, vertex group weights, normalization/lock |
| UV and texture | Multiple UV maps, unwrap, seam, pin, island pack, UDIM | Loop/corner UVs, images/video/sequences, color space, mapping, sampling, packed/external assets |
| Material and shader | All standard shader nodes, Principled BSDF, surface/volume/displacement, node groups | Emission, transmission, SSS, coat, alpha, normal/bump, texture links, all sockets/flags/material slots |
| Armature and rig | Bone hierarchy, edit/rest/pose, bone collections, custom shapes, envelope, weight skinning | Bone ID/roll/inheritance/deform, bind information, pose properties, standard Rigify rig generation/metadata |
| Constraint and driver | All standard Object/Bone Constraints, IK/spline IK, all driver types | Stack, target, influence, space, variable/expression, dependency graph, cycle/execution dependency diagnostics |
| Shape key | Relative/absolute, vertex group, evaluation time | Key block, basis, value/order/slider range, animation/driver, mesh correspondence |
| Animation | Action/slot/channel, F-Curve, keyframe, NLA, timeline marker, retiming | Interpolation, handle, extrapolation, curve Modifier, blend/strip, subframe, fps/fps_base |
| Physics | Rigid body, cloth, soft body, fluid, particle/hair, force field, collision, dynamic paint, simulation nodes | System settings, solver/seed, frame range, dependencies, initial state, bake/point cache |
| Grease Pencil | Stroke/fill/layer/frame/attribute, draw/sculpt/paint, animation | Material, pressure/opacity/curve type, mask, Modifier, visual effect, geometry/line style |
| Camera, light, and world | Perspective/orthographic/panorama, DOF, stereo, all light types, probe, HDRI | Focal length/sensor/shift/clip, energy/color/shape/shadow, world nodes, unit |
| Rendering | PBR, path tracing, real-time rendering, volume, hair, motion blur, denoise, Freestyle | Scene render settings, sample/seed/device, pass/AOV, Cryptomatte, render layer, transparent film |
| Color and output | OpenColorIO, view transform, exposure, gamma, image/sequence/video/audio | OCIO config, display/view/look, linear HDR, PNG/JPEG/TIFF/OpenEXR, alpha/bit depth |
| Compositor | All standard compositing nodes, node groups, masks, multi-layer/pass compositing | Socket, link, Scene/View Layer reference, color/depth/normal/motion, file output |
| Tracking and mask | 2D/plane tracking, camera/object solve, lens distortion, roto mask | Movie Clip, track, marker, solve settings/results, mask spline/feather/animation |
| Video Sequencer | image/movie/sound/scene/meta strips, transition, effect, retiming, mix | Strip hierarchy/channel/offset/duration, Modifier, proxy, audio volume/pan/pitch, codec |
| Asset and library | Link/Append, Library Override, asset library/catalog, dependency resolution, pack/unpack | Linked ID, source library, override operation, ID remapping, resource URI/hash/license/author |
| Scripting and extension | Text Data-Block, custom property, registered extensions, scripted driver | Source text, dependency version, RNA identifier. Execution requires a declared runtime/permission |
| Input and output | .blend, GLB/glTF, USD/USDZ, Alembic, FBX, OBJ, PLY, STL, BVH, SVG, PDF | Format-specific conversion/axes/units/animation/assets/loss report. Implemented formats are specified in Section 12 |


In potter's current physics evaluation, in addition to rigid bodies, deterministic cloth and soft body with vertex-group pin/goal and collisions, seeded face/vertex particles, boundary-constrained SPH liquid with metaball surfaces, and dynamic-paint vertex color/weight driven by brush proximity are evaluated and cached per frame, and output to `bake --kind simulation`. Particles are evaluated as point markers, object instances, or hair strand meshes. Fluid smoke/fire and simulation nodes are reported as `not_supported`.

Each row lists representative items. Catalog all standard Modifiers, Constraints, and node types from the RNA/official API of the fixed Blender build, and compare names, properties, sockets, enums, defaults, read-only conditions, and required context mechanically. Omitted catalog entries, unimplemented variants, and insufficient native evaluation mean the first release has not met its requirements.

### Feature Catalog Contract

- Register a stable `feature_id`, supported Blender types, pot operations, schema, dependencies, and verification fixtures for each feature.
- Report capabilities individually for `create`, `edit`, `inspect`, `evaluate`, `render`, `import`, and `export`.
- Read-only computed values are inspect/evaluate targets. Do not count non-writable values as an editing gap. Make writable creation properties editable through typed operations.
- Replace creation operations that require GUI context with explicit targets, frames, coordinates, selection sets, and numeric parameters. Do not depend on the state of a running Blender UI.
- Make registrations available through `pot schema --kind capabilities --json` and `pot inspect --features --json`.
- Standard features in the table MUST NOT be considered supported solely through preservation-only blobs or generic Python calls.

## 2. Compatibility Items Added or Changed from the Original Specification

| Old Constraint | New Specification |
| --- | --- |
| One static Scene | Multiple Scenes, frame/subframe, View Layers, Collections |
| Only shape and material inside Object | Data-Block separation, shared geometry, multiple material slots, library |
| Only positive-scale TRS | Negative/zero scale, matrix/shear, all rotation modes, parent inverse |
| Only groups can be parents | Object, bone, and vertex parenting |
| Store triangles only | Preserve polygon topology, loose geometry, and corner attributes as canonical data. Triangulate during evaluation |
| Only vertex/face indices | Distinguish persistent element IDs from indices within an evaluation snapshot |
| alpha=1, fixed Lambert shading | All shaders, textures, volumes, transparency, color management, separate solid and beauty modes |
| Shared materials only | All Data-Block sharing, instances, make single user, subtree duplication |
| PNG plus object ID only | Frame, camera, View Layer, depth, normal, instance/element, render pass |
| GLB export only | Editable .blend round trips and import/export of standard exchange formats |
| Restore everything from scene.json alone | scene.json plus assets/binary/compat payloads referenced by content hash |
| Topology and reference checks only | Compatibility checks for the entire dependency graph, animation, rig, simulation, shader, library, and output |
| Fixed limit of 1,000,000 triangles | Configurable limits, chunk/stream processing, support for brushes, textures, volumes, and caches |
| Byte-for-byte match for all output | Strict reproduction of canonical hashes and ID buffers; distinguish semantic comparisons for rendering, solvers, and .blend |

## 3. Meaning of .blend Compatibility

.blend compatibility means round-tripping editable creation data, not merely "a mesh that opens."

- In `Blender → import → potter edit → export → Blender`, preserve all features, settings, and references except those being changed.
- In `potter → export → Blender edit → import → potter`, preserve fixed IDs, sharing relationships, hierarchy, and animation.
- Do not arbitrarily apply/bake/convert Modifiers, node graphs, rigs, Actions, or physics settings to meshes.
- Preserve all Scenes, hidden Objects, unused Data-Blocks, Text, custom properties, and workspaces. Also verify Blender's load/save conversions by diff.
- Byte-for-byte identity of .blend files and assets is not required. Verify semantic equivalence of names, types, references, sharing, numeric values, topology, graphs, animation, and settings.
- Output all structures to .blend. Do not reuse the display-oriented subset output used for GLB, etc.

Blender itself has version compatibility limitations. Treat round trips on the baseline version separately from migrations from older versions. [Blend File Compatibility](https://developer.blender.org/docs/handbook/guidelines/compatibility_handling_for_blend_files/)

### Version and External Dependencies

- Check the input writer version and minimum reader version. A newer version outside the supported profile returns `BLENDER_VERSION_UNSUPPORTED`.
- Use a verified migration profile for older versions. Record changed features, data, names, and values in a migration report. By default, fail migrations that cannot be returned to an equivalent state without changes.
- Register textures, fonts, sounds, Movie Clips, volumes, caches, and linked libraries in the dependency list. Preserve original path, storage URI, hash, retrieval status, and packed status.
- Resolve relative paths against the location of the original .blend. Use each library's own base path for that library. Do not use the import destination's cwd as the base.
- Collect all assets recursively. Preserve sharing, cycles, and library override relationships in ID references. Allow linked libraries to be saved as separate files.
- Output a sidecar bundle for types that cannot be packed into .blend. Do not incorrectly report them as being inside a single file.
- Report required execution dependencies such as extensions, scripts, and drivers. Missing dependencies cause evaluation/render errors; preservation on save is still possible. Do not claim compatibility when a standard feature's runtime is missing.
- Automatic execution of code in scene files is disabled by default. Execute required drivers/extensions in an authorized runtime, and include its settings in the evaluation hash.

### Unknown Data and Loss

- Preserve unknown Data-Blocks, properties, nodes, and custom payloads with their types, references, and original data.
- Preserve original data during a .blend no-op round trip. Verify reference diffs to determine whether unknown data can be retained when exporting after edits.
- Mark items that can only be preserved as `preserve_only`, distinct from items that can be edited/evaluated. The first release MUST NOT ship with any standard feature left as `preserve_only`.
- If an edit referencing unknown data cannot preserve its meaning, return `UNSUPPORTED_FEATURE`/`UNREPRESENTABLE_FEATURE`. Do not silently delete or flatten it.
- Distinguish retention targets defined in the format catalog from standard conversions such as evaluation and triangulation. List losses of retention targets in `losses` with `feature_id`, `data_id`, `reason`, and a proposed solution. Reject by default; perform an explicitly specified conversion only when `--allow-lossy` is provided. The entire creation graph in .blend is a retention target.
- Do not turn failures caused by missing runtimes, assets, or extensions into successes with `--allow-lossy`.

## 4. Import and Export Verification Fixtures

Prepare "create," "edit," "evaluate," and "round trip" fixtures for each feature category. The first release MUST NOT be considered complete until all of the following pass.

1. Catalog consistency for all Data-Block types, Modifiers, Constraints, and standard nodes/sockets. Include dependencies of bundled standard features.
2. .blend no-op round trip: preserve all Scenes, hidden items, orphans, fake users, shared Data-Blocks, names, and custom properties.
3. Edit round trip: no changes outside the specified targets. Modifiers, Geometry Nodes, shaders, rigs, animation, simulation, compositor, and VSE are editable.
4. Reimport after Blender edits: reuse fixed IDs. Detect additions, removals, renames, single-user changes, and library overrides as diffs.
5. Restore in a separate directory a bundle containing links, relative paths, packed textures, UDIM, fonts, sounds, Movie Clips, VDB, and caches.
6. Compare evaluated world matrices, meshes, normals, UVs, bone poses, skinning, shape keys, drivers, and Constraints across multiple frames/subframes.
7. Verify textures, metal, transmission, SSS, hair, volumes, motion blur, color management, and render passes with CPU rendering.
8. Reproduce sculpting, paint, and weight results by replaying the Brush world-space sample sequence and seed.
9. Reproduce bakes with fixed physics initial state, seed, and solver version. Invalidate old caches when settings change.
10. Verify editing and output of Grease Pencil, tracking, masks, compositor, and VSE, including audio synchronization.
11. Compare supported features, axes, units, and loss reports for GLB/glTF, USD/USDZ, Alembic, FBX, OBJ, PLY, STL, BVH, SVG, and PDF.
12. Distinguish corruption, versions that are too new, unknown extensions, missing assets, and unrepresentable formats with errors. Preserve existing canonical data and output when import/export fails.
13. Preserve shared Data-Blocks, libraries, rigs, Actions, simulations, and assets through undo/redo and restore after restart. Do not garbage-collect dependencies of past snapshots.

Comparison tolerances: for transform/shape lengths, `1e-6 m + abs(value) × 1e-6`; for angles, `1e-5 rad`; for weights, UVs, etc., `1e-6`. Names, IDs, references, topology, enums, booleans, and graph links MUST match exactly. For rendering and simulation, specify the seed, solver, color space, error metric, and threshold for each fixture; do not pass a fixture without a defined threshold.

## 5. References and Tracking

- Creation feature baseline: [Blender 5.2 Manual](https://docs.blender.org/manual/en/5.2/index.html).
- Data-Block, library, and format baseline: [Assets, Files, & Data System](https://docs.blender.org/manual/en/5.2/files/index.html).
- File version limitations: [Blend File Compatibility](https://developer.blender.org/docs/handbook/guidelines/compatibility_handling_for_blend_files/).
- Exchange format feature categories: [Blender Pipeline](https://www.blender.org/features/pipeline/).

When updating the baseline build, reverify catalog diffs, schema diffs, migration, and all fixtures. Do not claim automatic support for future features.
