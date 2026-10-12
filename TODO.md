# TODO

## pot workflow refine

- [ ] Next release: confirm the `webp` crate (static libwebp built with `cc`) builds for every dist target. PR CI builds only on Linux x86_64 (`ubuntu-latest`) and it was built locally on macOS aarch64; the Windows (x86_64/aarch64 msvc) and Linux aarch64 artifacts are first built by the release workflow.
- [ ] Grease Pencil strokes are drawn by the preview but `pot inspect` reports null bounds for them (`crates/potter-core/src/eval.rs`), so the floating-part check skips them.
