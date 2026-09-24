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
It runs `lake build VeriTile VeriTileFull`, rejects Lean `sorry` warnings, runs
the official comparator on every proven library target in the manifest, checks declared axioms
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

## Shared comparator gate

The artifact checker, `bench/check_ports.sh`, and `bench/audit_trust.sh` all
require the official comparator and the sandbox tools installed above. Missing
tools, export failures, and comparator rejections fail the gate. CI installs
the same pinned tools through `.github/actions/setup-comparator`.

The public Bench audit workflow first runs one global job: the full library
build, structural checks, and library comparator. Four dependent jobs build
the lite `VeriTile` target and audit disjoint shards of the standalone corpus.
Each runner keeps its own mutable comparator workspace. The final
`TritonBench-G audit complete` check requires both the global job and every
shard to succeed; failed, cancelled, or skipped dependencies fail that check.

Local `bench/audit_tritonbench_g.sh` runs still cover all gates and the whole
corpus by default. To reproduce the CI stages separately:

```bash
bench/audit_tritonbench_g.sh --global-only
AUDIT_TRUST_SHARD_COUNT=4 AUDIT_TRUST_SHARD_INDEX=0 bench/audit_trust.sh
```

Shard indices are 0–3; run every index for a complete corpus audit. Both shard
variables are required, and sharding cannot be combined with named targets or
`--global-only`. A global-only result or a single shard is not a completed
audit. The existing sharded aggregate command also remains available: setting
the two variables on `bench/audit_tritonbench_g.sh` runs the global gates and
the selected shard together.

```bash
python3 scripts/check_comparator.py --library
python3 scripts/check_comparator.py --file bench/examples/VectorAdd.lean --trust
python3 scripts/test_check_comparator.py
```

Routine checks freeze the current trusted project sources and build cache in
an independent temporary workspace. Comparator exports and replays the current
proofs against that source snapshot (`source-replay` mode). This is not a check
that statements stayed unchanged from a previous Git revision; specification
changes still require review. `prove.sh` retains its separate comparison against
the original task captured before the agent runs.

The library gate selects `proven` library rows from the manifest. Standalone
gates enumerate every theorem object retained in the file's Lean environment,
including private and macro-generated names, and expose checked aliases to
comparator. Definition-only fixtures report zero theorem targets and run a
separate trivial sentinel through comparator; this does not establish a kernel
correctness claim. Anonymous `example` commands that Lean does not retain as
theorem declarations are compilation tests, not inventoried proof targets.

The trust gate also retains its headline, statement, and circular-spec checks.
The aggregate bench audit performs compilation and comparator replay in one
trust pass. Batch workers share one independent snapshot, with separate output
modules, and concurrency accounts for export/replay memory. Raw `lake build`
remains the build step; use these gates for proof acceptance.

The structural rules in `bench/audit_tritonbench_g.sh` share source extraction
through `bench/audit_source.py`. Kernel selection and call formatting are explicit
options: checks covering every JIT helper must not silently switch to the
primary-kernel policy. These text scans complement the Lean/comparator gates;
they do not discover proof obligations. Run their parser and failure regressions
with `python3 -m unittest discover -s scripts -p 'test_audit_source.py'`.

In the same checkout, run `test_audit_gates.py` and a full corpus audit
sequentially. The regression suite creates deliberately invalid temporary ports
under `bench/tritonbench_g/`, which a concurrent full scan would also select.

Logs under `Logs/comparator-check-*` record source hashes, selected targets,
generated source, comparator configuration, diagnostics, binary hash, and exit
status. Temporary build-cache copies are removed when a batch finishes.
