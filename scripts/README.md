# Automatic proof attempts

`scripts/prove.sh` runs the `lean4` Claude Code plugin's `/lean4:autoprove`
command, then uses [Lean's official comparator](https://github.com/leanprover/comparator)
to decide whether the requested theorems were proved.

## Setup

Use Linux with Landlock support and a working systemd user service. Install
Go 1.24+, the project's Lean toolchain, and the Claude Code `lean4` plugin.
Build the judge tools once, outside the project (to avoid copying them into each
judging workspace):

```bash
scripts/setup-comparator.sh /tmp/veritile-proof-tools
export PATH="/tmp/veritile-proof-tools/bin:$PATH"
lake build VeriTile VeriTileFull
```

The installer pins upstream comparator commit
`2a00b30df5e9173e70c4e4ec669fdf03da3163b9`, which targets Lean 4.29.0,
its lockfile's lean4export revision, and landrun commit
`5283024a2f49b28046c3b4a06d7d775c058d4d80`. Revisit these pins when upgrading
Lean. The landrun pin preserves the CLI argument handling expected by this
comparator version. `COMPARATOR_BIN` may specify another compatible official comparator
executable; `landrun` and `lean4export` must be on `PATH`.

## Usage

```bash
scripts/prove.sh path/to/Task.lean --theorem MyKernel.correctness
scripts/prove.sh path/to/Task.lean --theorem MyKernel.correctness \
  --theorem MyKernel.equivalence --max-cycles 20 \
  --prompt "Try induction on n first."
```

Replace the schematic path and names with your task. The file must be inside
this project. **`--theorem` is required** and repeatable: the caller selects the
fully qualified declarations to judge, before the agent runs. The default
limit is five proof cycles. Only those selected theorems and their dependencies
are judged; this is not a claim that every declaration in the file is proved.

## Judgment and logs

Before proof search, the runner copies the original file, project Lean sources,
Lake configuration, and dependency/build cache into an independent temporary
workspace. This needs temporary disk space for `.lake` (reflinks are used where
supported). After the agent exits, only its resulting source file is admitted
to that workspace. The agent's build artifacts and changes to imports or
configuration are not copied back into the judge.

Comparator builds and compares `ComparatorChallenge` and `ComparatorSolution`,
checks the selected statements and their referenced definitions, checks for
unpermitted axioms, and replays the exported proof in Lean's kernel. The allowed
axioms are exactly `propext`, `Quot.sound`, and `Classical.choice`. The runner uses
upstream's systemd AF_UNIX restriction along with comparator's landrun sandbox.
It never falls back to matching agent text or compiler output.

- Exit 0: official comparator accepted all selected theorems.
- Nonzero: rejection, missing tools, unavailable sandbox, invalid arguments, or
  another execution failure. An agent's success report cannot override this.

Each attempt writes `Logs/<basename>_<unique-id>/` with the original challenge,
candidate source, comparator configuration and output, agent JSON stream,
input-source hashes, and `result.json` containing the two exit codes and the
comparator binary hash. The temporary copy of the build cache is removed.

The starting sources, dependencies, tools, and host must be trusted. This
wrapper runs a local coding agent with filesystem access; it does not isolate
a hostile agent from the host. Use separate solver/judge machines for that
threat model. Comparator compares declaration names exactly: private or
module-generated names in a statement can differ between the two modules and
be rejected even for a valid proof. Prefer public definitions for task statements
and their specification/kernel dependencies.

## Regression checks

With the judge tools on `PATH`, run:

```bash
python3 scripts/test_prove.py
```

These tests use the real comparator and a deterministic fake agent (no API
calls). They cover a valid proof, changed statements and definitions, extra
axioms, remaining `sorry`, deleted targets, and multiple requested theorems.

Proof search still depends on the installed `lean4` plugin version; record it
when reporting benchmark results. The previous benchmark setup used version
4.4.9 of the plugin.

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
