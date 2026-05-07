# VeriTile Reference Documentation

Long-form reference docs for VeriTile's embedded Triton subset, semantic
model, and trusted-boundary policies. These are *contracts*: each describes
what is modeled, what is not, and where the boundary sits. Historical proposals
and research-problem notes are also colocated here under their PascalCase names
(see [`Proposal.md`](./Proposal.md), [`ResearchProblemPointerRegion.md`](./ResearchProblemPointerRegion.md), etc.).

## Triton DSL surface

- [`TritonSubset.md`](./TritonSubset.md) — supported Triton-like surface
  syntax, semantic model, and the gaps that are not yet modeled. The
  authoritative artifact contract for the embedded DSL.
  - [`TritonSubset_zh.md`](./TritonSubset_zh.md) — Chinese version.

## Algorithm / Compute layer architecture

- [`CorrectnessSurfaces.md`](./CorrectnessSurfaces.md) — user-facing theorem
  surfaces for single-kernel correctness and two-kernel refinement:
  `ComputeCorrect.Output*`, `ComputeRefine.Output*Eq`, and their escape
  hatches.
- [`EraseDType.md`](./EraseDType.md) — the two-layer architecture:
  `ComputeKernel` (compute-facing AST, preserves bitcast / fp width
  spellings) projected to algorithm-layer `Kernel` via
  `ComputeKernel.toAlgorithm?` / `eraseDType`.
- [`KernelManifest.md`](./KernelManifest.md) — the schema for
  `scripts/kernel-manifest.tsv`, the canonical per-kernel registry used by
  artifact checks and future benchmark ports.

## Memory model

- [`GpuMemoryModel.md`](./GpuMemoryModel.md) — what "GPU memory" means in
  VeriTile: a functional-correctness model over `BlockState.mem`,
  `RegFile`, and `TensorView` metadata. Not a microarchitectural CUDA
  model.
- [`MemorySafety.md`](./MemorySafety.md) — Layer-1 bounds-safety predicate
  (`Op.MemorySafe` / `Stmt.MemorySafe` / `Kernel.MemorySafe` /
  `ComputeKernel.MemorySafe`) and the active-lane convention for masked
  loads / stores.

## Concurrency boundary

- [`ConcurrencySemantics.md`](./ConcurrencySemantics.md) — the current
  deterministic boundary for non-sequential GPU effects: where atomic /
  async / barrier primitives sit relative to the proof-facing semantics
  today, and what extensions are scoped for the long-term roadmap.

## Project plan

The end-to-end project plan and roadmap live in the repo root:

- [`../PLAN.md`](../PLAN.md) — English
- [`../PLAN_zh.md`](../PLAN_zh.md) — Chinese
