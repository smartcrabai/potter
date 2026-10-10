# TODO

## jcode upstream PRs

Worktrees: `../jcode-pr-1775` (branch `sdk-stop-daemon-on-failed-launch`), `../jcode-pr-1774` (branch `sdk-disable-instance-auto-update`); push to the `fork` remote (takumi3488/jcode).

- [ ] [#1776](https://github.com/1jehuang/jcode/pull/1776) (closes #1775): check CI and Greptile's re-review after the `4417326` follow-up (pre-launch daemon pid check, test-owned stand-in daemon); answer maintainer review.
- [ ] [#1777](https://github.com/1jehuang/jcode/pull/1777) (closes #1774): check CI and Greptile's re-review after the `30b3755` follow-up (`--no-update` on the bridge); answer maintainer review.
- [ ] CI failures in Format, Quality Guardrails, and Build & Test on Linux and macOS come from master `04c7d2b` (already noted in a comment on both PRs). Rebase if master fixes them and a maintainer asks for green CI.
- [ ] After merge: bump the `jcode-sdk` rev in `Cargo.toml` `[workspace.dependencies]` and remove the `JCODE_NO_AUTO_UPDATE` override in `crates/potter-workflow/src/refine/mod.rs` if the SDK now covers it; then delete both worktrees and branches.

## pot workflow refine

- [ ] Known gap until #1777 is released: the `JCODE_NO_AUTO_UPDATE` override does not stop updates when `jcode` is a source build (only `--no-update` on the bridge does, and the SDK fixes the bridge argv).

## Release

- [ ] Only debug builds were verified locally for `x86_64`/`aarch64` Linux and `x86_64`/`aarch64` Windows (the macOS `dist build` artifact was verified). Confirm the optimized `dist` builds on the first release CI run.
