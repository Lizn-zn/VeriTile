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
scripts/prove.sh path/to/hard.lean --max-cycles=20

# With a strategy hint
scripts/prove.sh path/to/file.lean --prompt "Try induction on n first."
```

## Exit codes

- 0 — `/lean4:autoprove` reported success (heuristically: clean exit, result subtype isn't "error", result text doesn't mention failure)
- 1 — failure (cycles exhausted, stuck, claude error, or argument error)

Logs are written to `Logs/<basename>_<timestamp>.json` for inspection / debugging.

## Pinned plugin version

The wrapper's behavior depends on the installed `lean4` Claude Code plugin version.

As of 2026-04-26 we use the version under `~/.claude/plugins/cache/lean4-skills/lean4/4.4.9/`.
If results stop being reproducible, check the plugin version first.

## What this is NOT

- Not a custom prover — all proof search happens inside `/lean4:autoprove`
- Not connected to lake directly — the plugin handles `lake env lean` internally
- Not Anthropic API directly — uses `claude -p` (Claude Code CLI) as the entry point

See `PLAN.md` decision log entry 5 for why this is a wrapper rather than a custom Python tool.

## Artifact checker

`scripts/check-artifact.sh` is the local release/CI gate for the Lean artifact.
It runs `lake build`, rejects Lean `sorry` warnings, checks declared axioms
against `scripts/artifact-axiom-whitelist.txt`, verifies the key theorem surface
listed in `scripts/artifact-theorems.txt`, and checks the examples manifest /
README example links for drift.

```bash
scripts/check-artifact.sh
```
