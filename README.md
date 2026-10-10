# potter

The name potter means "a person who makes objects from clay." It is a headless, AI-friendly 3D creation tool that aims to be a Blender alternative.

- [Specification](docs/spec.md)
- [Blender Compatibility Requirements](docs/blender-compatibility.md)
- [Glossary](GLOSSARY.md)
- [Skill for AI](skills/potter/SKILL.md)
- [Changelog](CHANGELOG.md)

Normal processing uses potter's own engine. Headless Blender 5.2.x is used only for .blend import/export.

## Install

macOS and Linux:

```sh
brew install smartcrabai/tap/potter
# or
curl --proto '=https' --tlsv1.2 -LsSf https://github.com/smartcrabai/potter/releases/latest/download/potter-installer.sh | sh
```

- Windows: download `potter-x86_64-pc-windows-msvc.zip` or `potter-aarch64-pc-windows-msvc.zip` from the [latest release](https://github.com/smartcrabai/potter/releases/latest) and put `pot.exe` on `PATH`.
- From source: `cargo install --git https://github.com/smartcrabai/potter --locked` (CI builds with Rust 1.99, pinned in `rust-toolchain.toml`).

Optional dependencies:

- Blender 5.2.x, only for `.blend` import/export. Found via `--blender <path>`, `POTTER_BLENDER`, `PATH`, then `/Applications/Blender.app`.
- `jcode` on `PATH` with a provider login, only for `pot workflow`.

## Usage

```sh
pot init ./scene --json
pot apply ./scene --file ops.json --preview iso --json
pot schema --kind capabilities --json   # List of supported/unsupported features (not_supported includes a reason)
```

The authoritative sources for supported and unsupported features are `pot schema --kind capabilities --json` and `pot inspect <scene> --features --json`.

## Agent workflows (`pot workflow`)

`pot workflow` runs multi-agent workflows on top of potter, driven by [jcode](https://github.com/1jehuang/jcode) agents through `jcode-sdk`. It needs `jcode` on PATH with a provider login.

```sh
pot workflow -h                           # list available workflows
pot workflow refine -p openai -m gpt-5.5 -i chair.jpg -i chair-side.jpg
```

### `refine`

A modeler agent builds a model from the reference images (`-i/--image`, repeatable), potter renders it from several views, a fresh reviewer agent lists what blocks acceptance, and the modeler repairs exactly those findings. The loop is built to converge:

- The reviewer reports only blocking findings: render problems, missing or extra parts, wrong basic shape, proportions off by roughly 15% or more, misplaced or detached parts, and clearly wrong colors. Smaller differences, shading or shadow artifacts, and unseen sides are recorded as `non_findings` and never block.
- Findings keep stable IDs (`F1`, `F2`, …). Each follow-up review sees the previous review and the modeler's fix report, must mark every previous finding `resolved` or `persists`, may not raise decided non-findings again, and adds new findings only for blocking problems such as regressions. The modeler may report a finding as `blocked` instead of faking a fix, and the reviewer can accept that as a non-finding.
- A deterministic geometry check passes mesh parts that touch neither the ground nor the grounded assembly, with their gap in meters, to the reviewer.
- The loop stops as `converged` (no findings, exit 0), `stalled` (the finding count did not beat the best review for `--stall-limit` consecutive reviews, default 2), or `max_iterations_reached` (`--max-iterations`, default 5); both non-converged statuses exit 2.

Output goes to `-o/--out` (default `<first image stem>-refine`, numbered when taken): the potter project in `<out>/scene` and each iteration's renders, `review.json`, and `fix-report.json` in `<out>/iter-NN/`. stdout is one JSON line with `status`, `iterations`, and `remaining_findings`; with `--json` it is a `pot` response envelope carrying that summary as `result`, and errors become `INTERNAL_ERROR` envelopes.

`-p/--provider <NAME>` and `-m/--model <ID>` are required and pick the jcode provider and model for the modeler and every reviewer; they are sent to jcode as `<NAME>:<ID>`, and the provider needs a jcode login.

## Repository layout

| Path | Package | Role |
| --- | --- | --- |
| `src/`, `tests/` | `potter` | the `pot` binary (argument parsing, `--json` envelopes) and its end-to-end tests |
| `crates/potter-core/` | `potter-core` | engine library: scene model, evaluation, rendering, exchange, and the typed command layer |
| `crates/potter-workflow/` | `potter-workflow` | `pot workflow` agent workflows (jcode) |

## Support

If you find potter useful, consider [sponsoring smartcrabai](https://github.com/sponsors/smartcrabai)
to support its development and maintenance.

## License

potter is licensed under the GNU General Public License v3.0 or later ([LICENSE](LICENSE)). Parts of the modifier, constraint, animation, and remesh evaluators are derived from [Blender](https://www.blender.org) (GPL-2.0-or-later) and [OpenVDB](https://www.openvdb.org) (Apache-2.0); affected source files name their upstream origin and copyright holders in their headers.
