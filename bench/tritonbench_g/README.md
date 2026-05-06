# TritonBench-G v1 source kernels

Upstream Python kernels from
[TritonBench-G v1](https://github.com/thunlp/TritonBench/tree/main/data/TritonBench_G_v1)
that VeriTile is in the process of porting and verifying.

This directory holds the `.py` source files only. Lean ports of each kernel
live under `VeriTile/Examples/` (typical layout: `VeriTile/Examples/TritonBenchG/<KernelName>.lean`).

## Layout

```
bench/tritonbench_g_v1/
├── README.md      (this file)
└── <kernel>.py    one file per kernel, filename preserved from upstream
```

## Provenance

| Date imported | Upstream commit | Notes |
|---|---|---|
| (TBD) | (TBD) | Initial import |

When importing a new batch, append a row and pin the upstream commit so future
imports remain reproducible.

## Licensing

Before importing any kernel, confirm `thunlp/TritonBench` is under an
import-compatible license (MIT / Apache 2 / BSD). Vendoring under an
incompatible license (e.g. GPL) would impose downstream constraints on
VeriTile. Record the upstream LICENSE text alongside the imported files.

## Adding a kernel

1. Pick a file from upstream `data/TritonBench_G_v1/`.
2. Cross-check its verdict in [`../tritonbench_coverage.md`](../tritonbench_coverage.md). Prefer `OK` first; `Soft` next; `Hard` only when the relevant capability extension has landed.
3. Copy the `.py` file here, preserving the upstream filename.
4. Lean-port it under `VeriTile/Examples/...` following the example-file layout and `documents/TheoremSurfaces.md` naming convention.
5. Add a manifest entry in `scripts/kernel-manifest.tsv` with `source = tritonbench:<filename>` and `source_ref = <upstream commit>`.
6. Update the Provenance table above if importing from a fresh upstream commit.

## See also

- [`../tritonbench_coverage.md`](../tritonbench_coverage.md) — static coverage classification
- [`../README.md`](../README.md) — overall benchmark policy
- [`../../documents/KernelManifest.md`](../../documents/KernelManifest.md) — manifest schema
