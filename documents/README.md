# VeriTile Reference Documentation

**English** | [中文](README_zh.md)

Long-form reference for VeriTile's embedded Triton subset, semantic model,
and trusted-boundary policies. Each active doc is a *contract*: what is
modeled, what is not, where the boundary sits.

Closed-phase design notes and resolved research-problem records live in
[`archive/`](./archive/) — kept as historical artifacts; not load-bearing
for current work.

## Active Reference Docs

Task-oriented index (mirrors the top-level [`README.md`](../README.md)
Documentation Map):

| Question | Doc |
|---|---|
| Which Triton constructs are supported? | [TritonSubset.md](./TritonSubset.md) ([中文](./TritonSubset_zh.md)) |
| How does Tilelang map onto the neutral core? | [TilelangMapping.md](./TilelangMapping.md) |
| Where does my new lemma / definition belong? | [CodeOrganization.md](./CodeOrganization.md) ([中文](./CodeOrganization_zh.md)) |
| Tactic conventions (incl. `erw` carrier-bridge) | [ProofConventions.md](./ProofConventions.md) ([中文](./ProofConventions_zh.md)) |
| Which theorem surface should I use? | [CorrectnessSurfaces.md](./CorrectnessSurfaces.md) |
| Naming conventions for theorem surfaces | [TheoremSurfaces.md](./TheoremSurfaces.md) |
| How does the kernel manifest work? | [KernelManifest.md](./KernelManifest.md) |
| How does dtype erasure work? | [EraseDType.md](./EraseDType.md) |
| How does memory safety / framing work? | [MemorySafety.md](./MemorySafety.md) |
| What's the GPU memory model? | [GpuMemoryModel.md](./GpuMemoryModel.md) |
| How are atomics / async copies modeled? | [ConcurrencySemantics.md](./ConcurrencySemantics.md) |
| ApproxGeLU midrange certified-error strategy | [ApproxGeluPhiStrategy.md](./ApproxGeluPhiStrategy.md) |

## Architecture / Roadmap

The end-to-end project plan and roadmap live in the repo root:

- [`../PLAN.md`](../PLAN.md) — architecture, decision log, phase status
- Live roadmap: GitHub issue
  [`#91`](https://github.com/Lizn-zn/VeriTile/issues/91)

## Archive

Closed-phase notes are preserved in [`archive/`](./archive/):

- `DslMacroOptions.md` — macro design exploration, decision: option C
  (`triton { ... }` block macro). Closed.
- `ForLoopInvDesign.md` — Phase B `forLoop_inv` interface decisions.
  Closed; current implementation matches.
- `ResearchProblemPointerRegion.md` — RP1: pointers vs named regions.
  Resolved: named regions throughout.
- `ResearchProblemAddressTyping.md` — RP2: ℝ-uniform vs Nat-bifurcated
  `Value`. Resolved: bifurcated `Value`.
- `Proposal.md` / `Proposal_zh.md` — initial project proposal. The
  technical content has evolved beyond it; kept for historical context.
