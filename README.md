# potter

The name potter means "a person who makes objects from clay." It is a headless, AI-friendly 3D creation tool that aims to be a Blender alternative.

- [Specification](docs/spec.md)
- [Blender Compatibility Requirements](docs/blender-compatibility.md)
- [Glossary](GLOSSARY.md)
- [Skill for AI](skills/potter/SKILL.md)
- [Changelog](CHANGELOG.md)

Normal processing uses potter's own engine. Headless Blender 5.2.x is used only for .blend import/export.

```sh
cargo build --release   # target/release/pot
pot init ./scene --json
pot apply ./scene --file ops.json --preview iso --json
pot schema --kind capabilities --json   # List of supported/unsupported features (not_supported includes a reason)
```

The authoritative sources for supported and unsupported features are `pot schema --kind capabilities --json` and `pot inspect <scene> --features --json`.
