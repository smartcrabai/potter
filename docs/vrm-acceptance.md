# VRM Support Targets and Acceptance

Potter is responsible for authoring and export fidelity. Tracking capture, retargeting, lip-sync input, and the render loop belong to the consuming application.

## Supported targets

| VRM | Target | Status |
| --- | --- | --- |
| 1.0 (`--format vrm`) | `@pixiv/three-vrm` 3.5.5 + `three` 0.180.0 in Playwright 1.64 Chromium, with the consumer fix in `tests/vrm-consumer/patches/` | Automated (`tests/vrm-consumer`, CI job `vrm-consumer`). No VRM 1.0 desktop application is certified |
| 0.x (`--format vrm0`) | Same SDK through its VRM 0.x path (`VRMUtils.rotateVRM0`) | Automated |
| 0.x (`--format vrm0`) | VSeeFace v1.13.38c5, Windows 8 or later, 64-bit | Manual procedure below. **Not yet run** |

VSeeFace supports VRM 0.x only. Do not give it a VRM 1.0 file.

### Consumer fixes

The harness patches `@pixiv/three-vrm` 3.5.5 (`tests/vrm-consumer/patches/`) where it deviates from the VRM 1.0 specification. Potter's output follows the specification; the exporter does not compensate.

- `materialColorBinds` / `textureTransformBinds` are applied once per mesh that uses the material, so a material shared by N meshes moves N times. The patch binds each material instance once.
- Bone look-at selects `rangeMapVerticalUp` when looking down and `rangeMapVerticalDown` when looking up. The patch selects the maps as `lookAt.md` defines.

Known unpatched gap: `VRMUtils.rotateVRM0` rotates the scene but not VRM 0.x world `gravityDir`, so sideways spring gravity in a VRM 0.x file points the opposite way in three-vrm. Potter writes `gravityDir` the way UniVRM migrates it (VRM 0.x `(x, y, z)` = VRM 1.0 `(-x, y, z)`), and the harness asserts three-vrm's raw value.

## Automated acceptance

`cd tests/vrm-consumer && npm ci && npm test` (set `POT` to choose the `pot` binary; `POTTER_SWIFTSHADER=1` forces software WebGL, which Linux always uses).

1. Builds a representative avatar only through the `pot` CLI: generated human rig, eye bones, continuous skinned limbs with 2–3 blended bone weights, smooth UV-sphere head with UV seams and two material slots, 11 shape keys, bone-parented hat sharing a material with the torso, a single-sided ribbon and its mirrored copy, MToon with textures, normal map, outline, UV-animation mask, alpha clip and double-sided blend.
2. Exports VRM 1.0 (full), VRM 1.0 (expression look-at), and VRM 0.x, and loads only the `.vrm` files.
3. Khronos glTF Validator: 0 errors. Warnings must match `validate/warning-allowlist.json` (code, pointer, reason); new or stale entries fail. Official VRM JSON schemas at a pinned, hash-locked commit.
4. Chromium + three-vrm checks with exact expectations: upright grounded rest; posed skinning against Potter's evaluated mesh; morphs at weights 0, 0.25, 0.5, 1, 0; shared-material color/UV binds applied once; combined expressions with `isBinary` and `block` / `blend` overrides; bone and expression look-at; spring sphere and capsule collisions, reset, varying time step, finite state and bone length; smooth vs flat shading, outline, alpha holes, double-sided, mirrored faces, normal map, UV-animation mask, per-face materials; VRM 0.x parity with VRM 1.0.

## Manual acceptance: VSeeFace (VRM 0.x)

Prerequisites: Windows 8+ 64-bit, VSeeFace v1.13.38c5 from the official site, a webcam, and optionally a microphone.

1. Run `npm test` in `tests/vrm-consumer`, then copy `.cache/fixture/v0.vrm` (or your own `pot export --format vrm0` output) to the Windows machine. Copy only the `.vrm` file.
2. Load it as a local avatar. Pass: loads without errors; textures and thumbnail visible.
3. Rest: the avatar faces the camera, stands upright, and its feet touch the floor.
4. Head and body: turn, nod and tilt your head; lean left and right. Pass: the avatar follows on the same side without lag, jitter or reversed axes.
5. Face: blink, open your mouth, and speak the vowels A, I, U, E, O. Pass: eyelids close fully without passing through the face; each mouth shape matches.
6. Gaze: look left, right, up and down with your eyes only. Pass: the eyes follow without reversed axes and stay inside the eye sockets.
7. Expressions: trigger the avatar's emotion expressions (the harness fixture has `joy` and `sorrow`) through VSeeFace's expression hotkeys. Pass: each looks right and returns to neutral.
8. Spring bones: shake your head. Pass: the hair swings, settles, and does not pass through the head.
9. Rendering: rotate the light (Ctrl + drag). Pass: smooth shading on the head, outlines, transparent and double-sided parts, and per-face materials (eyes vs skin) match the authored settings.
10. Tracking loss: cover the camera for 10 seconds, then uncover it. Pass: tracking recovers and the pose returns to normal.
11. Reload: load another avatar, then load this one again. Pass: same result as steps 2–9.
12. Endurance: run for at least 60 minutes with normal movement. Pass: no bone drift, no material or expression residue, no crash.

Record the date, Windows version, VSeeFace version, `.vrm` file hash (`pot export` reports it), and pass/fail per step with screenshots of failures.
