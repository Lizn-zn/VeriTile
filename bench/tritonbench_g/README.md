# TritonBench-G v1 ports

Workspace for VeriTile's port of [TritonBench-G v1](https://github.com/thunlp/TritonBench/tree/main/data/TritonBench_G_v1) — 184 GitHub-scraped real Triton kernels released as the headline channel of TritonBench (ACL 2025 Findings).

This directory holds one subdirectory per kernel. Each subdirectory bundles **upstream Python source + VeriTile DSL port + a per-kernel README**, so a single port lives entirely under one folder.

## Layout

```
bench/tritonbench_g/
├── README.md                       (this file)   # coverage table: ../tritonbench_coverage.md
└── <kernel_name>/
    ├── README.md                   per-kernel notes (status, gotchas, TODO)
    ├── <kernel_name>.py            upstream Python source (pinned, see Provenance)
    └── <KernelName>.lean           VeriTile DSL port; namespace `VeriTile.Bench.TritonBenchG.<KernelName>`
```

The Lean filename is the **CamelCase form** of the directory name (e.g. `vector_addition/` contains `VectorAddition.lean`). The namespace mirrors that — no `.Port` suffix or other padding.

## Status interpretation

A port goes through three stages, tracked per-kernel in `README.md`:

1. **DSL port** — `<KernelName>.lean` follows the **faithful 1:1 transcription** contract for the upstream `.py` kernel in `triton { ... }` syntax. Allowed mechanical Lean-syntax changes are documented in [`review_criteria.md`](./review_criteria.md); explicit translation exceptions are registered in [`completion_audit.md`](./completion_audit.md#remaining-blockers). The port may not compile if it uses DSL surface that has not yet landed — failing-to-compile is the intended signal that the DSL surface needs extension. **Inventory: 173 Python/Lean port pairs; 11 of the 184 work directories are README-only scaffolds.** Compilation and audit evidence is described below.
2. **Spec** — Real-valued mathematical specification of the kernel's intended output is written.
3. **Verification** — `ComputeCorrect.Realizes` / `ComputeRefine.Realizes` theorem is proved and registered in `scripts/kernel-manifest.tsv`.

Stage 1 is the verbatim transcription contract; reaching stage 3 (verification) requires both the DSL gap to close and a proof to land.

## Current audit state

The current sweep is tracked in [`completion_audit.md`](./completion_audit.md).
`bench/check_ports.sh` compiles every Python/Lean port pair and requires
official comparator replay. The aggregate audit also checks comment-stripped
sources for placeholder proofs. Completed CI evidence is pinned to a specific
run and commit in [`coverage-ci.json`](../../site/src/lib/coverage-ci.json);
it does not certify later revisions or pending runs. A `null` record means
no completed public-repository corpus audit has been recorded yet.

There are no current explicit algorithm-layer `hAlg` blockers. Any future
proof blockers should be listed in [`proof_blockers.md`](./proof_blockers.md).
The stronger #146 proof-status audit is tracked in
[`proof_gap_manifest.tsv`](./proof_gap_manifest.tsv) and checked by
[`../check_proof_gap_manifest.py`](../check_proof_gap_manifest.py). That
manifest covers all 345 headline declarations across 173 ports. The explicit
2026-09-24 review records 8 `full_value_candidate`, 296 `specialization`,
37 `precomputed_input_slice`, 3 `pre_rounding_slice`, and 1 `blocked_summary`
rows. [`coverage_review.json`](./coverage_review.json) records each headline's
actual scope and checks it against Lean/Python source fingerprints and Python
function links. Explicit source annotations remain binding; source changes
require a new review. A candidate label does not certify GPU execution, and
historical issue links do not imply that the issue is currently open or that
closing it completed a proof.

## Build

These ports are intentionally **not** part of the main library glob in `lakefile.toml`. They live alongside the upstream sources in the benchmark workspace. To compile them:

```bash
# all currently-ported kernels
bench/check_ports.sh

# mechanical audit gates for the current TritonBench-G sweep
bench/audit_tritonbench_g.sh

# reviewed scope classification for every headline declaration
python3 bench/check_proof_gap_manifest.py

# subset by kernel name
bench/check_ports.sh vector_addition softmax_triton1
```

The port-check script elaborates each `<KernelName>.lean` and requires official
comparator export/replay, reports per-kernel pass/fail, and exits non-zero on
any failure. The aggregate audit combines compilation, Lean trust/statement
checks, and comparator replay in `bench/audit_trust.sh`, avoiding a duplicate
port-build pass. The public CI workflow partitions the standalone corpus across
four GitHub-hosted jobs; all four must pass. Local runs cover the whole corpus
unless `AUDIT_TRUST_SHARD_COUNT` and `AUDIT_TRUST_SHARD_INDEX` are explicitly set
(see [`scripts/README.md`](../../scripts/README.md#shared-comparator-gate)).
Its other gates include Python/Lean count matching,
placeholder-proof scanning, correctness-surface scanning, compiled-port README
status checks, and a documented-scope check for Python `.to(tl.float32)` casts
that are outside a Lean proof slice. It also rejects Lean-only
`tl.load(..., dtype=...)` annotations and `keep_dims` reduction substitutions,
and it requires Python `+=` updates missing from Lean to be covered by a
documented slice or branch/surface scope, including a normalized left-hand-side
check for pointer names such as `a_ptr` versus `A`. Upstream `rsqrt` calls are
also checked for preservation, and Lean-only `tl.where` statements are
rejected. The same audit compares Python and Lean `tl.*(...)` call surfaces and
requires any missing or extra call to be covered by an explicit
slice/specialization note, and it compares `for` / `while` / `if` counts inside
the Python `@triton.jit` kernel body and Lean `triton { ... }` body. Ordered
`tl.*(...)` call sequences are checked as well, so unannotated call reordering
is rejected mechanically. Top-level statement left-hand-side sequences are also
checked, including `+=`, annotated assignments, and tuple assignments.
It also checks that `proof_gap_manifest.tsv` is fresh against the Lean source,
so newly added or renamed `output_summary` declarations cannot bypass the #146
proof-status classification.
It is a mechanical gate only; line-by-line faithfulness still follows
[`review_criteria.md`](./review_criteria.md), and unresolved proof obligations
remain tracked in [`proof_blockers.md`](./proof_blockers.md).

## Provenance

| Date imported | Upstream commit | Kernels | Notes |
|---|---|---|---|
| 2026-05-06 | [`603e28a`](https://github.com/thunlp/TritonBench/commit/603e28a) | 15 (Tier 1) | initial DSL ports; no specs / theorems yet |
| 2026-05-13 | [`603e28a`](https://github.com/thunlp/TritonBench/commit/603e28a) | 141 port pairs | the audited port set as of that date; see `completion_audit.md` for remaining proof obligations |
| 2026-07-04 | [`603e28a`](https://github.com/thunlp/TritonBench/commit/603e28a) | 1 (`reversed_cumsum_scalar`) | reverse-range port; DSL blocker resolved by #94/#448 |
| 2026-08-25 | [`603e28a`](https://github.com/thunlp/TritonBench/commit/603e28a) | 173 port pairs | current audited port set (`int_scaled_matmul` closed the integer family); the remaining 11 of the 184 upstream kernels are README-only scaffolds |

### Local modifications to vendored `.py` files

The vendored `.py` files are **not** strictly byte-identical to upstream. The following modifications may be applied locally to imported files:

- **Input type annotations on every `@triton.jit` kernel signature.** Pointer args annotated `tl.tensor`, runtime int scalars `tl.int32`, runtime float scalars `tl.float32`. `tl.constexpr` annotations from upstream are preserved as-is. These annotations are JIT-equivalent (Triton ignores non-`constexpr` Python type hints at compile time), so kernel behavior is unchanged — they exist purely as in-source documentation that aligns Python signatures with the type information the Lean ports rely on.

When importing a new batch:

1. Pin the upstream commit you fetched from in this table.
2. Ensure the upstream LICENSE has not changed (currently **Apache-2.0**).
3. Add per-file attribution headers in each `.py` (see [Licensing](#licensing) below).
4. Apply the input-type annotations described above to each `@triton.jit` signature.

## Licensing

Upstream `thunlp/TritonBench` is licensed under **Apache-2.0**. VeriTile is MIT-licensed; Apache-2.0 → MIT vendoring is permitted with attribution.

Each vendored `.py` file should carry an attribution header similar to:

```python
# Source: thunlp/TritonBench@<commit-hash>
#   data/TritonBench_G_v1/<filename>.py
# Upstream license: Apache-2.0 (see https://github.com/thunlp/TritonBench/blob/main/LICENSE)
```

The initial imports landed without these headers (commit `eab9b81`); back-filling them is open work.

## Adding a kernel

1. Pick a file from upstream `data/TritonBench_G_v1/`.
2. Cross-check its verdict in [`../tritonbench_coverage.md`](../tritonbench_coverage.md). Prefer `portable now`; a `blocked` kernel only after the primitive it names has landed (with a consumer).
3. Create `bench/tritonbench_g/<kernel_name>/`.
4. Drop the upstream `.py` in there with the attribution header above.
5. Add `<KernelName>.lean` with the DSL port. Use `namespace VeriTile.Bench.TritonBenchG.<KernelName>` and import `VeriTile.Triton.Core` + `VeriTile.Triton.DSL`.
6. Verify with `bench/check_ports.sh <kernel_name>`.
7. (Stage 2/3) When you write a spec / proof, add a `scripts/kernel-manifest.tsv` row with `source = tritonbench:<filename>.py` and `source_ref = <upstream-commit>`.
8. Update the Provenance table above if importing from a fresh upstream commit.

## See also

- [`../tritonbench_coverage.md`](../tritonbench_coverage.md) — measured coverage classification across all 184 kernels
- [`../README.md`](../README.md) — overall benchmark policy
- [`../check_ports.sh`](../check_ports.sh) — port build script
- [`../../documents/KernelManifest.md`](../../documents/KernelManifest.md) — manifest schema (used at stage 3)
- [`../../documents/TheoremSurfaces.md`](../../documents/TheoremSurfaces.md) — naming convention for verification theorems
