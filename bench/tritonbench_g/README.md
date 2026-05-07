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

1. **DSL port** — `<KernelName>.lean` is a **faithful 1:1 transcription** of the upstream `.py` kernel into `triton { ... }` syntax. Allowed mechanical Lean-syntax changes only: `=` → `:=`, pointer args → `RegionName` injected via `$(...)`, `tl.constexpr` annotation → Lean `Nat`/`Bool` parameter, Lean scalar params → `$(...)`. The port may not compile if it uses DSL surface that has not yet landed — failing-to-compile is the intended signal that the DSL surface needs extension. **Compiles today: 15 / 15.**
2. **Spec** — Real-valued mathematical specification of the kernel's intended output is written.
3. **Verification** — `ComputeKernel.ComputeCorrect` / `ComputeKernel.ComputeRefine` theorem is proved and registered in `scripts/kernel-manifest.tsv`.

Stage 1 is the verbatim transcription contract; reaching stage 3 (verification) requires both the DSL gap to close and a proof to land.

## DSL gaps blocking faithful ports

Distinct surface gaps surfaced by the current 15 transcriptions (each port's docstring also records its own reason). Closing any one of these unblocks the ports listed alongside it:

| Gap | Affected ports |
|---|---|
| `tl.program_id(axis=0)` keyword form | `add_example`, `add_value`, `sin_computation`, `triton_mul2`, `vector_addition` |
| `tl.max(x, 0)` positional-axis form | `logsumexp_fwd` |
| `tl.max(..., return_indices=True)` tuple return + multi-binding `a, b := ...` | `max_reduction` (third kernel only; first two compile) |
| Real literal `0` / `0.0` written directly inside `tl.where` / arithmetic | `relu_triton_kernel` |
| `tl.math.*` (libdevice) namespace | `sin_kernel` |
| `(x).to(tl.float32)` algorithm-layer cast | `cosine_compute` |

Fixing these is L3 operator-coverage work; track under #15 / #86 surfaced gaps. Re-run `bench/check_ports.sh` after each gap closes to flip the affected ports green.

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

### Local modifications to vendored `.py` files

The vendored `.py` files are **not** strictly byte-identical to upstream. The following modifications are applied locally to all 15 imports:

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
