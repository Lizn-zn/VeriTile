---
title: Reference
description: Build instructions, scripts, and external resources for working with VeriTile.
---

Operational reference. Most of these point back to the repository itself
since they describe the build and tooling rather than the design.

## Build & toolchain

- **Lean toolchain**: `v4.29.0`. Pinned in
  [`lean-toolchain`](https://github.com/Lizn-zn/VeriTile/blob/main/lean-toolchain).
- **Build**: `lake build` from the repo root.
- **Manifest + sorry check**: `scripts/check-artifact.sh`.
- **Bench port check**: `bench/check_ports.sh`.

## Sub-project READMEs

These live in the repo and are the authoritative source for their respective
areas:

- [Top-level README](https://github.com/Lizn-zn/VeriTile/blob/main/README.md)
  — quick-start + theorem-surface chooser.
- [`bench/tritonbench_g/` README](https://github.com/Lizn-zn/VeriTile/blob/main/bench/tritonbench_g/README.md)
  — bench layout, port checklist, kernel inventory.
- [`scripts/` README](https://github.com/Lizn-zn/VeriTile/blob/main/scripts/README.md)
  — what each script does and how CI uses them.
- [`verso/` README](https://github.com/Lizn-zn/VeriTile/blob/main/verso/README.md)
  — the architecture-overview slide deck (separate sub-project).
- [`documents/` index](https://github.com/Lizn-zn/VeriTile/blob/main/documents/README.md)
  — the original markdown design notes (this site re-renders them under
  [Architecture](/VeriTile/architecture/) and [Proofs](/VeriTile/proofs/)).
