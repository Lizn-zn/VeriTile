# scripts/prove.sh

Thin wrapper around the `lean4` Claude Code plugin's `/lean4:autoprove` command.
Used for VeriTile's LLM benchmark eval (see `PLAN.md` §LLM benchmark protocol).

## Usage

```bash
scripts/prove.sh <lean_file> [--max-cycles N] [--prompt "extra text"]
```

Examples:

```bash
# Try to close all sorries in the held-out file with up to 5 cycles
scripts/prove.sh bench/llm_eval/softmax_naive_correct_held_out.lean

# More aggressive search
scripts/prove.sh path/to/hard.lean --max-cycles 20

# With a strategy hint
scripts/prove.sh path/to/file.lean --prompt "Try induction on n first."
```

## Exit codes

- 0 — Claude exited successfully, the reported result subtype is not `error`,
  and an independent `lake env lean` check passed with no `sorry` or `error`
  text in its output.
- 1 — the attempt or final Lean check failed, or the arguments were invalid.

Logs are written to `Logs/<basename>_<timestamp>.json` for inspection / debugging.
If the attempt fails, the final Lean diagnostics are also saved to
`Logs/<basename>_<timestamp>_leancheck.txt`. The default limit is five proof
cycles. Artifact and axiom audits are separate commands; the wrapper does not
run them automatically.

## Pinned plugin version

The wrapper's behavior depends on the installed `lean4` Claude Code plugin version.

As of 2026-04-26 we use the version under `~/.claude/plugins/cache/lean4-skills/lean4/4.4.9/`.
If results stop being reproducible, check the plugin version first.

## Execution

- Proof search happens inside `/lean4:autoprove`.
- The wrapper rechecks the resulting file with `lake env lean` after the plugin exits.
- The entry point is `claude -p` (Claude Code CLI).

See `PLAN.md` decision log entry 5 for why this is a wrapper rather than a custom Python tool.

## Artifact checker

`scripts/check-artifact.sh` is the local release/CI gate for the Lean artifact.
It runs `lake build VeriTile VeriTileFull`, rejects Lean `sorry` warnings, checks declared axioms
against `scripts/artifact-axiom-whitelist.txt`, validates the per-kernel
registry in `scripts/kernel-manifest.tsv`, and checks README example links for
drift. It also resolves documented public API names with Lean through
`site/scripts/check-doc-api.py`.

`scripts/kernel-manifest.tsv` is the source of truth for public kernel/example
metadata: file, theorem symbol, theorem kind, verification status, source,
static config, label, and notes. See `documents/KernelManifest.md` before
adding a new public example or benchmark port.

```bash
scripts/check-artifact.sh
```
