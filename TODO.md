# TODO

## jcode upstream PRs

Worktrees: `../jcode-pr-1775` (branch `sdk-stop-daemon-on-failed-launch`), `../jcode-pr-1774` (branch `sdk-disable-instance-auto-update`); push to the `fork` remote (takumi3488/jcode).

- [ ] [#1776](https://github.com/1jehuang/jcode/pull/1776) (closes #1775) and [#1777](https://github.com/1jehuang/jcode/pull/1777) (closes #1774): Greptile's findings are fixed and its re-review scores both 5/5; answer maintainer review.
- [ ] CI failures in Format, Quality Guardrails, and Build & Test on Linux and macOS come from jcode master (still failing at `02ae3ed`; noted in a comment on both PRs). Rebase if master fixes them and a maintainer asks for green CI.
- [ ] After merge: bump the `jcode-sdk` rev in `Cargo.toml` `[workspace.dependencies]` and remove the `JCODE_NO_AUTO_UPDATE` override in `crates/potter-workflow/src/refine/mod.rs` if the SDK now covers it; then delete both worktrees and branches.

## pot workflow refine

- [ ] Known gap until #1777 is released: the `JCODE_NO_AUTO_UPDATE` override does not stop updates when `jcode` is a source build (only `--no-update` on the bridge does, and the SDK fixes the bridge argv).

## Ideas (not decided)

- Count `stalled` by whether previous findings get resolved instead of by the finding count: a run that fixed one finding per review still stopped as `stalled`.
- Reference image size is not checked; large photos can exceed provider image limits.
- The floating-part check covers meshes only, not curves, text, or other geometry.
