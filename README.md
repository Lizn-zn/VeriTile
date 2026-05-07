# VeriTile

**English** | [中文](README_zh.md)

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
- **Theorem surfaces**: `ComputeCorrect.Output*` for kernel ↔ math
  specification, `ComputeRefine.Output*Eq` for kernel pair equivalence.
  Both project through `toAlgorithm?` and run on `Kernel.Correct` /
  `Kernel.Refine` underneath.
- **Examples**: 15 ported TritonBench-G kernels with proofs (see
  [`bench/tritonbench_g/`](./bench/tritonbench_g/)) plus FlashAttention-1
  forward, online softmax, Welford, LayerNorm, log-sum-exp.
- **CI gate**: `lake build` + `scripts/check-artifact.sh` (no `sorry`,
  axiom whitelist, manifest schema, doc-drift checks) +
  `bench/check_ports.sh`.

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
| One kernel matches a 1D `Fin n → ℝ` math spec | `ComputeCorrect.OutputArray` |
| One kernel matches an ND tensor spec | `ComputeCorrect.OutputTile` |
| One kernel matches a scalar reduction | `ComputeCorrect.OutputScalar` |
| Value + index output (e.g. `tl.max(..., return_indices=True)`) | `ComputeCorrect.OutputPairWhere` |
| Two kernels produce equal observable outputs | `ComputeRefine.OutputArrayEq` (or `OutputTileEq`) |
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
    ComputeCorrect.OutputArray
      (addKernel xReg yReg outReg n) s
      (programTileView s outReg n)
      (fun i => xs i + ys i) := by
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
  Triton/                  Core AST, semantics, memory, DSL, math
  Examples/                Worked correctness/refinement proofs
bench/tritonbench_g/       TritonBench-G v1 ports (15 closed)
documents/                 Design notes, subset spec, surface guide
scripts/                   CI gate, kernel manifest, LLM proof wrapper
verso/                     Slide deck / overview
```

## Verification

- `lake build` — typecheck + build full library and examples
- `scripts/check-artifact.sh` — `lake build` ∧ `no sorry` ∧ axiom
  whitelist ∧ kernel-manifest schema ∧ README/doc-term drift
- `bench/check_ports.sh` — per-port build of the TritonBench-G ports

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
