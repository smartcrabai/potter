# potter

A headless, AI-friendly 3D creation tool. It handles modeling, sculpting, rigging, and animation, and aims to be a Blender alternative.

## Language

**potter**:
The product name means "a person who makes objects from clay."

**Scene**:
The target of 3D creation work. It is where components are edited and inspected, previews are generated, and models are exported.

**Operation**:
An editing unit applied to a scene. Multiple operations can be applied together.

**Preview**:
An image of a scene viewed from a specified direction for review.

**Project**:
The working directory passed to the `pot` command. It holds multiple scenes, creation data, assets, and Blender compatibility information.

**Data-Block**:
Creation data such as shapes, materials, and Actions. It is separate from Objects and can be shared by multiple Objects or Scenes.

**Snapshot**:
An observation/evaluation target pinned to a revision, Scene, View Layer, frame, and dependency hashes.

**Blender Compatibility**:
A contract for editing and evaluating the standard creation features of the baseline version, and preserving creation structure, sharing, and dependencies through .blend import/export.

**Blend Adapter**:
A conversion layer that uses Headless Blender only to read and write .blend files. Normal processing uses potter's own engine.
