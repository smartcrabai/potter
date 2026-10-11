# TODO

## jcode SDK

- [x] Adopt the fixes from [#1776](https://github.com/1jehuang/jcode/pull/1776) and [#1777](https://github.com/1jehuang/jcode/pull/1777). Both PRs were closed without a GitHub merge; the maintainer landed them directly on master with Takumi's authorship (`109aa923d`, `59466b209`, `8775959ee`, `282784809`). Potter pins the final fix at `282784809686e353353bdf680b39a2a5bce6804e`.
- [x] Remove the local `JCODE_NO_AUTO_UPDATE` override: SDK private launches now set it and pass `--no-update` to the bridge, covering source builds too. Failed launches stop only daemons they started.
- [ ] Clean up the old contribution worktrees/branches separately when authorized: `../jcode-pr-1775` (`sdk-stop-daemon-on-failed-launch`) and `../jcode-pr-1774` (`sdk-disable-instance-auto-update`).

## Ideas (not decided)

- Count `stalled` by whether previous findings get resolved instead of by the finding count: a run that fixed one finding per review still stopped as `stalled`.
- Reference image size is not checked; large photos can exceed provider image limits.
- The floating-part check covers meshes only, not curves, text, or other geometry.
