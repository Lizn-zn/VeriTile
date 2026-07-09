# VeriTile Code Organization

**English** | [中文](CodeOrganization_zh.md)

VeriTile separates three concerns into three layers. Knowing which layer
something belongs in saves churn when adding new operators, new bridge
lemmas, and new kernel transcriptions.

## The three layers

```
                    ┌─────────────────────────────────────────────┐
                    │  bench/tritonbench_g/<kernel>/X.lean        │  ← per-kernel glue
   per-kernel       │  VeriTile/Examples/X.lean                   │
                    │  - kernel definition (`triton { ... }`)     │
                    │  - kernel-specific *Spec, *Load, *Offset    │
                    │  - kernel correctness theorem               │
                    └─────────────────────────────────────────────┘
                                       │ uses
                                       ▼
                    ┌─────────────────────────────────────────────┐
                    │  VeriTile/Triton/Semantics/X.lean           │  ← bridging mechanisms
    bridging        │  - `Tile`, `WithBot`, `Option.map₂`         │
                    │  - tiled indexing (lane, validLanes, ...)   │
                    │  - masked reductions (sup', sum, dot)       │
                    │  - kernel-agnostic, but Triton-bound        │
                    └─────────────────────────────────────────────┘
                                       │ uses
                                       ▼
                    ┌─────────────────────────────────────────────┐
                    │  VeriTile/Triton/Math/X.lean                │  ← pure math
   pure math        │  - `(Fin N → ℝ) → ...` operators            │
                    │  - non-trivial mathematical identities      │
                    │  - depends only on Mathlib                  │
                    └─────────────────────────────────────────────┘
```

Each layer can only depend on the layers below it.

## What goes where

### `VeriTile/Triton/Math/`

**Naming**: by mathematical operator. `Math/Activation.lean`,
`Math/Reduction.lean`, `Math/L2Norm.lean`, `Math/LogSumExp.lean`,
`Math/Softmax.lean`, `Math/Loss.lean`.

**Belongs here**:
- A definition that takes `Fin N → ℝ` (or `ℝ → ℝ`, etc.) and returns a real,
  with no `BlockState`, `RegionName`, `WithBot`, or `Tile` in the signature.
- An identity over those operators (e.g.
  `naive_softmax = stable_softmax`, `welford_running = twoPass`).

**Inclusion rule**: at least 2 callers (or one caller plus an imminent
second), and the definition is mathematically substantive (not a 1-line
argument-binding wrapper).

**Does not belong here**:
- Kernel layout (`s.pid * stride + i.val`)
- `WithBot` / `Tile` / `Option.map₂` — those go to `Semantics/`
- Per-kernel `*Spec` that just binds the math to a kernel's argument tuple

### `VeriTile/Triton/Semantics/`

**Naming**: by **mechanism**, not by operator. `Semantics/TiledIndexing.lean`,
`Semantics/MaskedReduction.lean`, `Semantics/Step.lean`,
`Semantics/State.lean`. (Future candidates: `StreamingAccumulator`,
`AtomicReduction`, `BroadcastReshape`.)

A mechanism is "the way `Tile` / `WithBot` / `Option.map₂` interact with a
class of math operators." Each `Semantics/X.lean` file owns one mechanism
and stays kernel-agnostic.

**Belongs here**:
- Lane-and-block indexing primitives, partition lemmas
  (`laneIdx`, `validLanes`, `blockIndex`, `sum_exp_partition`).
- WithBot/Tile carrier-bridge lemmas
  (`withBot_sup'_partial`, `reduceSum_masked_sq_eq_some_sum`,
  `reduceSum_masked_dot_eq_some_sum`).
- Anything generic over `(load, active)` or `(load, mask)` that any
  L2-style / softmax-style / reduction-style kernel can plug into.

**Does not belong here**:
- A single kernel's specific `s.pid * stride` math — that's the next layer.
- Non-Triton math — that's `Math/`.
- A mechanism with only one kernel using it: keep it inline in the bench file
  until a second user appears (same "≥ 2 callers" rule as `Math/`).

**Why mechanism-named, not operator-named**: an operator (e.g. softmax)
typically uses several mechanisms (tiled indexing + masked reduction). A
mechanism (e.g. masked reduction) is reused across operators. Mechanism-named
files keep the deduplication right: `MaskedReduction.lean` holds bridges for
softmax / log-sum-exp / L2 / max-with-mask alike.

### `bench/tritonbench_g/<kernel>/`, `VeriTile/Examples/`

**Naming**: per-kernel (`L2NormTriton1.lean`, `FlashAttention1.lean`).

**Belongs here**:
- The `triton { ... }` kernel transcription
- Kernel-local helpers: `*Offset`, `*Load`, `*Carrier`, `*Spec`
- The kernel correctness theorem (algorithm-layer + compute-facing)

**Pattern**: glue code. The `*Spec` reads "this kernel writes
`Triton.TiledX.operator (load_xs ...) ...`" by binding kernel layout to
math operators through the bridge lemmas in `Semantics/`.

## How to add a new piece

When you write a new lemma or definition, ask:

1. **Does it touch `BlockState`, `RegionName`, or kernel layout?**
   → It's per-kernel glue. Put it in the kernel file.

2. **Does it touch `Tile` / `WithBot` / `Option.map₂` but is generic over
   the kernel's load function and mask predicate?**
   → It's a mechanism. Put it in the existing `Semantics/X.lean` whose
   theme matches, or open a new `Semantics/<MechanismName>.lean` once a
   second user appears.

3. **Is it purely `(Fin N → ℝ) → ...` and reused by ≥ 2 kernels?**
   → It's a math operator. Put it in `Math/X.lean` named after the operator.

4. **Is it a 1-line composite of Mathlib + the kernel's argument names?**
   → Keep it kernel-local. Don't move it to `Math/`.

## Physical module layout

The three-layer rule above is about *where new math/bridge/glue lives*. The
physical `VeriTile/Triton/` tree also carries the semantic, correctness, and
float infrastructure. Current directory map:

```text
VeriTile/
  Triton.lean               Umbrella prelude — `import VeriTile.Triton` pulls
                            the whole subset (Core + Semantics + Memory +
                            KernelLemmas + Correctness + Float + DSL + Math +
                            Launch + Concurrency). All 151 bench ports and the
                            showcase files import this single module.
  Triton/
    Core/                   AST: the `Kernel` / `ComputeKernel` types and the
                            `ComputeOp` bit constants (the former
                            `Triton/Compute.lean` was folded in here and deleted).
    Semantics/              Typed operational semantics: exec, step, tiled
                            indexing, masked reduction, streaming accumulator, …
    Memory/                 BlockState, tensor views, readback.
    DSL/                    `triton { ... }` macro front-end.
    Math/                   Pure `(Fin N → ℝ) → ...` operators (see three-layer
                            rule). Math/Erf is split: lightweight
                            `Triton.Math.Erf` (def `realErf`, used by
                            Semantics/TileOps) vs heavy `Math.RealErf` (the
                            Gaussian-integral proofs) — the split keeps the lite
                            build's Mathlib closure small.
    KernelLemmas/           Reusable bench-proof helpers (EvalHelpers,
                            LoopInvariant, Matmul, OffsetInjective, ScatterStore).
                            This directory was renamed from `Triton/Kernel/`; it
                            holds proof lemmas, NOT the `Kernel` type (that is in
                            Core/Ast).
    Correctness.lean        Top-level correctness/refinement surfaces:
                            `Kernel.Correct_without_Rounding`, `ComputeCorrect.*`,
                            `ComputeRefine.*`, `WriteMap`, `OutputReadable`.
                            (Moved out of Float/; was
                            `VeriTile.Triton.Float.Correctness`.)
    Float/                  Floating-dtype machinery only: dtype erasure
                            (Erasure, StateErasure) + the rounding model
                            (RoundingModel, EvalOpR, StepR, Refine, Pipeline).
                            The rounding surfaces `Realizes`/`Refines`/
                            `RefinesAt` live in `Float/Refine.lean`.
    Launch/                 Grid-launch composition: GridWriteFootprint,
                            GridFrames, mergeFrames, GridWritesDisjoint.
    Concurrency/            Grid-wide atomic-add correctness.
```

### Dependency direction

`Concurrency/Atomic` depends on `Launch.Composition` (it reuses
GridWriteFootprint / GridFrames / GridWritesDisjoint), and `Launch` never
imports `Concurrency` — so the edge is one-way, no cycle. Place
**Concurrency/Atomic *above* Launch** in any layer diagram: grid-wide
atomic-add correctness is an application layer built on top of grid-launch, not
below it.

## See also

- [`ProofConventions.md`](./ProofConventions.md) — proof-tactic conventions,
  including the `erw` carrier-bridge fallback.
- [`CorrectnessSurfaces.md`](./CorrectnessSurfaces.md) — the user-facing
  theorem surfaces (`Realizes`, `Refines`, `WriteMap`, `OutputReadable`).
