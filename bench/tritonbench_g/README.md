# TritonBench-G v1 ports

Workspace for VeriTile's port of [TritonBench-G v1](https://github.com/thunlp/TritonBench/tree/main/data/TritonBench_G_v1) — 184 GitHub-scraped real Triton kernels released as the headline channel of TritonBench (ACL 2025 Findings).

This directory holds one subdirectory per kernel. Each subdirectory bundles **upstream Python source + VeriTile DSL port + a per-kernel README**, so a single port lives entirely under one folder.

## Layout

```
bench/tritonbench_g/
├── README.md                       (this file)
├── tritonbench_coverage.md         (static coverage analysis, 184 kernels)
└── <kernel_name>/
    ├── README.md                   per-kernel notes (status, gotchas, TODO)
    ├── <kernel_name>.py            upstream Python source (pinned, see Provenance)
    └── <KernelName>.lean           VeriTile DSL port; namespace `VeriTile.Bench.TritonBenchG.<KernelName>`
```

The Lean filename is the **CamelCase form** of the directory name (e.g. `vector_addition/` contains `VectorAddition.lean`). The namespace mirrors that — no `.Port` suffix or other padding.

## Status interpretation

A port goes through three stages, tracked per-kernel in `README.md`:

1. **DSL port** — `<KernelName>.lean` is a **faithful 1:1 transcription** of the upstream `.py` kernel into `triton { ... }` syntax. Allowed mechanical Lean-syntax changes are documented in [`review_criteria.md`](./review_criteria.md). The port may not compile if it uses DSL surface that has not yet landed — failing-to-compile is the intended signal that the DSL surface needs extension. **Compiles today: 141 / 141 port pairs; 43 of the 184 work directories are README-only scaffolds and are not counted as completed ports.**
2. **Spec** — Real-valued mathematical specification of the kernel's intended output is written.
3. **Verification** — `ComputeCorrect.Realizes` / `ComputeRefine.Realizes` theorem is proved and registered in `scripts/kernel-manifest.tsv`.

Stage 1 is the verbatim transcription contract; reaching stage 3 (verification) requires both the DSL gap to close and a proof to land.

## Current audit state

The current sweep is tracked in [`completion_audit.md`](./completion_audit.md).
`bench/check_ports.sh` compiles every Python/Lean port pair and currently
reports `TritonBench-G ports: 141 ok, 0 fail`. The placeholder scan
`rg -n "True := by|trivial|sorry|admit" bench/tritonbench_g -g '*.lean'`
currently reports no matches.

The remaining non-green items are proof obligations, not DSL transcription
failures. They are listed in [`proof_blockers.md`](./proof_blockers.md):
`mean_reduction`, `embedding_triton_kernel`, and the forward real path of
`diag_ssm_triton` expose their algorithm-layer postconditions explicitly until
their loop invariants are connected to the concrete loop bodies.

## Build

These ports are intentionally **not** part of the main library glob in `lakefile.toml`. They live alongside the upstream sources in the benchmark workspace. To compile them:

```bash
# all currently-ported kernels
bench/check_ports.sh

# mechanical audit gates for the current TritonBench-G sweep
bench/audit_tritonbench_g.sh

# subset by kernel name
bench/check_ports.sh vector_addition softmax_triton1
```

The script runs `lake env lean` against each `<KernelName>.lean` independently, reports per-kernel pass/fail, and exits non-zero on any failure (CI-friendly).
The audit script wraps this port-build gate with Python/Lean count matching,
placeholder-proof scanning, correctness-surface scanning, compiled-port README
status checks, and a documented-scope check for Python `.to(tl.float32)` casts
that are outside a Lean proof slice. It also rejects Lean-only
`tl.load(..., dtype=...)` annotations and `keep_dims` reduction substitutions,
and it requires Python `+=` updates missing from Lean to be covered by a
documented slice or branch/surface scope, including a normalized left-hand-side
check for pointer names such as `a_ptr` versus `A`. Upstream `rsqrt` calls are
also checked for preservation, and Lean-only `tl.where` statements are
rejected.
It is a mechanical gate only; line-by-line faithfulness still follows
[`review_criteria.md`](./review_criteria.md), and unresolved proof obligations
remain tracked in [`proof_blockers.md`](./proof_blockers.md).

## Provenance

| Date imported | Upstream commit | Kernels | Notes |
|---|---|---|---|
| 2026-05-06 | [`603e28a`](https://github.com/thunlp/TritonBench/commit/603e28a) | 15 (Tier 1) | initial DSL ports; no specs / theorems yet |
| 2026-05-13 | [`603e28a`](https://github.com/thunlp/TritonBench/commit/603e28a) | 141 port pairs | current audited port set; see `completion_audit.md` for remaining proof obligations |

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
2. Cross-check its verdict in [`tritonbench_coverage.md`](./tritonbench_coverage.md). Prefer `OK` first; `Soft` next; `Hard` only when the relevant capability extension has landed.
3. Create `bench/tritonbench_g/<kernel_name>/`.
4. Drop the upstream `.py` in there with the attribution header above.
5. Add `<KernelName>.lean` with the DSL port. Use `namespace VeriTile.Bench.TritonBenchG.<KernelName>` and import `VeriTile.Triton.Core` + `VeriTile.Triton.DSL`.
6. Verify with `bench/check_ports.sh <kernel_name>`.
7. (Stage 2/3) When you write a spec / proof, add a `scripts/kernel-manifest.tsv` row with `source = tritonbench:<filename>.py` and `source_ref = <upstream-commit>`.
8. Update the Provenance table above if importing from a fresh upstream commit.

## See also

- [`tritonbench_coverage.md`](./tritonbench_coverage.md) — static coverage classification across all 184 kernels
- [`../README.md`](../README.md) — overall benchmark policy
- [`../check_ports.sh`](../check_ports.sh) — port build script
- [`../../documents/KernelManifest.md`](../../documents/KernelManifest.md) — manifest schema (used at stage 3)
- [`../../documents/TheoremSurfaces.md`](../../documents/TheoremSurfaces.md) — naming convention for verification theorems
