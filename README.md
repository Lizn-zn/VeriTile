# VeriTile

**English** | [中文](README_zh.md)

📖 **Docs site:** [lizn-zn.github.io/VeriTile/](https://lizn-zn.github.io/VeriTile/) (bench cookbook, status, architecture). Run locally: `./site/scripts/dev.sh`.

VeriTile embeds a typed Triton-style kernel DSL in Lean 4 and proves
correctness or refinement of those kernels against mathematical
specifications or against each other. The implementation embeds the kernel
language with `triton { ... }` syntax, defines an operational semantics over
typed `Op : TileDType → TileShape → Type` terms, and exposes
`ComputeCorrect` / `ComputeRefine` theorem surfaces for end users.

Algorithmic proofs run over the erased `.real` (mathematical) channel; the
optional `GapPolicy` records — but does not internally prove — the
compute-to-algorithm gap (IEEE-754 / PTX / TMA / concurrency), which stays
externally checked. See [Triton subset and gaps](./documents/TritonSubset.md).

## What VeriTile Does

- **DSL**: typed Triton subset with `triton { ... }` macro, ND tile shapes,
  reductions, masks (`mask=`/`other=`), block-pointer ops, and bare
  `if`/`for` control flow.
- **Theorem surfaces**: `ComputeCorrect.Realizes` for *one kernel vs a math
  specification*, `ComputeRefine.Refines` for *one kernel refining another*
  (writes-equality: the two final memories agree at every cell outside the
  declared scratch regions). Both project through `toAlgorithm?` and run on
  `Kernel.Correct_without_Rounding` / `Kernel.Refine` underneath.
- **Narrow-float / rounding-model layer** (#447): an abstract `RoundingModel`
  (`round : FloatDType → ℝ → ℝ`, sole axiom `round_real = id`) threads a
  black-box rounding function through the semantics (`evalOpR` / `stepStmtR` /
  `execR`). The unqualified surfaces are the rounding-parametric ones —
  `ComputeRefine.Realizes` (single kernel vs an R-annotated spec) and
  `Refines` / `RefinesAt` (two kernels) run under a `RoundingModel R`; the
  exact-ℝ idealization is the qualified `*_without_Rounding` name, which the
  bridge `Realizes.toRealizes` degenerates out to (as
  `ComputeCorrect.Realizes_without_Rounding`) at the trivial model. See the
  fused-vs-unfused SwiGLU showcase
  [`bench/examples/FusedSwiglu.lean`](./bench/examples/FusedSwiglu.lean).
- **Examples**: 151 ported TritonBench-G kernels with proofs (source of truth:
  [`bench/tritonbench_g/completion_audit.md`](./bench/tritonbench_g/completion_audit.md);
  see [`bench/tritonbench_g/`](./bench/tritonbench_g/)) plus FlashAttention-1
  forward, online softmax, Welford, LayerNorm, log-sum-exp.
- **CI gate**: `lake build` + `scripts/check-artifact.sh` (no `sorry`,
  axiom whitelist, manifest schema, doc-drift checks).
  `bench/check_ports.sh` is a separate local check (not run in CI) that
  builds the TritonBench-G ports one by one.

Out of scope: IEEE-754 floating-point semantics, PTX-level codegen,
detailed concurrency (atomics / async-copy serialization, beyond the
projection boundary), Python wrapper execution.

## Quick Start

### 1. Write a `ComputeKernel`

Kernels are region-polymorphic: memory regions arrive as `RegionName`
parameters; `tl.load` / `tl.store` use first-class pointer expressions.

```lean
def addKernel (xReg yReg outReg : RegionName) (n : Nat) : ComputeKernel := triton {
  pid     = tl.program_id(0)
  offsets = pid * $(n) + tl.arange(0, $(n))
  x       = tl.load($(xReg) + offsets)
  y       = tl.load($(yReg) + offsets)
  tl.store($(outReg) + offsets, x + y)
}
```

### 2. Choose a theorem surface

| Goal | Use |
|---|---|
| One kernel matches an output spec | `ComputeCorrect.Realizes` |
| One kernel refines another (writes-equality) | `ComputeRefine.Refines` |
| Two kernels agree pointwise per address | `ComputeRefine.RefinesAt` |
| Single kernel / pairs under a rounding model (narrow-float) | `ComputeRefine.Realizes` / `Refines` / `RefinesAt` |
| Value + index output (e.g. `tl.max(..., return_indices=True)`) | `ComputeCorrect.OutputPairWhere` |
| Custom postcondition over the final state | `ComputeCorrect.Post` / `ComputeRefine.Post` |
| Relation over arbitrary initial states (rare) | `ComputeCorrect.General` / `ComputeRefine.General` |

Full surface guide: [CorrectnessSurfaces.md](./documents/CorrectnessSurfaces.md).

### 3. Prove via the projected algorithm

```lean
theorem add_kernel_correct
    (xReg yReg outReg : RegionName) (n : Nat) (hN : 0 < n)
    (s : BlockState) (xs ys : Fin n → ℝ)
    (h_x : TensorView.loadedArray s (programTileView s xReg n) xs)
    (h_y : TensorView.loadedArray s (programTileView s yReg n) ys) :
    ComputeCorrect.Realizes_without_Rounding
      (kernel := addKernel xReg yReg outReg n)
      (initialState := s)
      (write := fun i : Fin n => some (outReg, s.pid * n + i.val))
      (expected := fun i => xs i + ys i) := by
  -- bridge to the projected algorithm kernel, then close on Real semantics
  ...
```

The standard pattern: `ComputeKernel.computeCorrect_of_toAlgKernel rfl`
discharges the projection, then `simp` reduces `exec` to the body
recurrence; the algebraic content closes by `simp` on the spec or by
invoking a math identity from `Mathlib`. The LLM proof wrapper
`scripts/prove.sh` automates this loop.

### 4. Register in the kernel manifest

Add a row to [`scripts/kernel-manifest.tsv`](./scripts/kernel-manifest.tsv)
so `scripts/check-artifact.sh` recognizes the theorem in CI. Schema and
naming conventions: [KernelManifest.md](./documents/KernelManifest.md),
[TheoremSurfaces.md](./documents/TheoremSurfaces.md).

## Minimal Example

Elementwise vector add against the `addSpec xs ys i = xs i + ys i` math
spec — see [`VeriTile/Examples/VectorAdd.lean`](./VeriTile/Examples/VectorAdd.lean).

## Refinement Example

Naive vs numerically-stable softmax (kernel pair refinement) — see
[`VeriTile/Examples/SoftmaxEq.lean`](./VeriTile/Examples/SoftmaxEq.lean).

## Documentation Map

Task-oriented:

| Question | Doc |
|---|---|
| Which Triton constructs are supported? | [TritonSubset.md](./documents/TritonSubset.md) |
| What semantic caveats affect theorem interpretation? | [SemanticCaveats.md](./documents/SemanticCaveats.md) |
| Where does my new lemma / definition belong? | [CodeOrganization.md](./documents/CodeOrganization.md) |
| Tactic conventions (incl. `erw` carrier-bridge) | [ProofConventions.md](./documents/ProofConventions.md) |
| Which theorem surface should I use? | [CorrectnessSurfaces.md](./documents/CorrectnessSurfaces.md) |
| How does dtype erasure work? | [EraseDType.md](./documents/EraseDType.md) |
| How does memory safety / framing work? | [MemorySafety.md](./documents/MemorySafety.md) |
| What's the GPU memory model? | [GpuMemoryModel.md](./documents/GpuMemoryModel.md) |
| How are atomics / async copies modeled? | [ConcurrencySemantics.md](./documents/ConcurrencySemantics.md) |
| How does the kernel manifest work? | [KernelManifest.md](./documents/KernelManifest.md) |
| Naming conventions for theorem surfaces | [TheoremSurfaces.md](./documents/TheoremSurfaces.md) |
| LLM proof wrapper | [scripts/README.md](./scripts/README.md) |

## Repository Layout

```text
VeriTile/
  Triton.lean              Umbrella prelude (`import VeriTile.Triton`)
  Triton/
    Core/                  AST (Kernel/ComputeKernel type, ComputeOp bits)
    Semantics/             Typed operational semantics (exec, execR)
    Memory/                BlockState, tensor views, readback
    DSL/                   `triton { ... }` front-end
    Math/                  Pure `(Fin N → ℝ) → ...` operators (+ Math/Erf)
    KernelLemmas/          Reusable bench-proof helpers (was Triton/Kernel/)
    Correctness.lean       Top-level correctness/refinement surfaces
                           (Kernel.Correct_without_Rounding, ComputeCorrect.*, ComputeRefine.*)
    Float/                 Floating-dtype machinery: dtype erasure +
                           the rounding model (RoundingModel, execR, Refine,
                           Pipeline)
    Launch/                Grid-launch composition / write footprints
    Concurrency/           Grid-wide atomic-add correctness (above Launch)
  Examples/                Worked correctness/refinement proofs
bench/tritonbench_g/       TritonBench-G v1 ports (151 pairs; see completion_audit.md)
bench/examples/            Showcase proofs (SwiGLU rounding invariance, ...)
documents/                 Design notes, subset spec, surface guide
scripts/                   CI gate, kernel manifest, LLM proof wrapper
verso/                     Slide deck / overview
```

## Verification

- `lake build` — typecheck + build full library and examples
- `scripts/check-artifact.sh` — `lake build` ∧ `no sorry` ∧ axiom
  whitelist ∧ kernel-manifest schema ∧ README/doc-term drift
- `bench/check_ports.sh` — per-port build of the TritonBench-G ports
  (local / manual; not part of the CI gate)

## Environment

- Lean 4 (`v4.29.0`) + Mathlib
- [Claude Code CLI](https://docs.claude.com/en/docs/claude-code) +
  [`lean4-skills`](https://github.com/lean4-skills/lean4-skills)
- `jq` (used by `scripts/prove.sh`)

## Roadmap

Long-running project. Goal: bring real Triton kernels (forward, backward,
concurrency, production layouts / autograd) into Lean's proof scope with
minimal modification. Live roadmap:
[#91](https://github.com/Lizn-zn/VeriTile/issues/91). Architecture and
decision log: [PLAN.md](./PLAN.md).

## License

[MIT](./LICENSE) © 2026 Zenan Li.
