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

1. **DSL port** — `<KernelName>.lean` defines the kernel via `triton { ... }` syntax and compiles. (Today: 15 / 184.)
2. **Spec** — Real-valued mathematical specification of the kernel's intended output is written.
3. **Verification** — `ComputeKernel.ComputeCorrect` / `ComputeKernel.ComputeRefine` theorem is proved and registered in `scripts/kernel-manifest.tsv`.

Stage 1 alone is a useful artifact (it confirms the DSL surface covers this kernel's primitives), but the project's verification claim only kicks in at stage 3.

## Build

These ports are intentionally **not** part of the main library glob in `lakefile.toml`. They live alongside the upstream sources in the benchmark workspace. To compile them:

```bash
# all currently-ported kernels
bench/check_ports.sh

# subset by kernel name
bench/check_ports.sh vector_addition softmax_triton1
```

The script runs `lake env lean` against each `<KernelName>.lean` independently, reports per-kernel pass/fail, and exits non-zero on any failure (CI-friendly).

## Provenance

| Date imported | Upstream commit | Kernels | Notes |
|---|---|---|---|
| 2026-05-06 | [`603e28a`](https://github.com/thunlp/TritonBench/commit/603e28a) | 15 (Tier 1) | initial DSL ports; no specs / theorems yet |

When importing a new batch:

1. Pin the upstream commit you fetched from in this table.
2. Ensure the upstream LICENSE has not changed (currently **Apache-2.0**).
3. Add per-file attribution headers in each `.py` (see [Licensing](#licensing) below).

## Licensing

Upstream `thunlp/TritonBench` is licensed under **Apache-2.0**. VeriTile is MIT-licensed; Apache-2.0 → MIT vendoring is permitted with attribution.

Each vendored `.py` file should carry an attribution header similar to:

```python
# Source: thunlp/TritonBench@<commit-hash>
#   data/TritonBench_G_v1/<filename>.py
# Upstream license: Apache-2.0 (see https://github.com/thunlp/TritonBench/blob/main/LICENSE)
```

The current 15 imports landed without these headers (commit `eab9b81`); back-filling them is open work.

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
