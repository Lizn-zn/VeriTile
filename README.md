# VeriTile

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
- **Theorem surfaces**: `ComputeCorrect.Realizes_without_Rounding` for *one kernel vs a math
  specification*, `ComputeRefine.Refines` for *one kernel refining another*
  (writes-equality: the two final memories agree at every cell outside the
  declared scratch regions). Both project through `toAlgorithm?` and run on
  `Kernel.Correct_without_Rounding` / `Kernel.Refine` underneath.
- **Narrow-float / rounding-model layer** (#1): an abstract `RoundingModel`
  (`round : FloatDType → ℝ → ℝ`, fields `round_real` (real-channel identity) and `round_idem` (idempotence)) threads a
  black-box rounding function through the semantics (`evalOpR` / `stepStmtR` /
  `execR`). The unqualified surfaces are the rounding-parametric ones —
  `ComputeRefine.Realizes` (single kernel vs an R-annotated spec) and
  `Refines` / `RefinesAt` (two kernels) run under a `RoundingModel R`; the
  exact-ℝ idealization is the qualified `*_without_Rounding` name, which the
  bridge `Realizes.toRealizes_without_Rounding` degenerates out to (as
  `ComputeCorrect.Realizes_without_Rounding`) at the trivial model. See the
  fused-vs-unfused SwiGLU showcase
  [`bench/examples/FusedSwigluEquiv.lean`](./bench/examples/FusedSwigluEquiv.lean).
- **Examples**: 173 ported TritonBench-G kernels with proofs (source of truth:
  [`bench/tritonbench_g/completion_audit.md`](./bench/tritonbench_g/completion_audit.md);
  see the [per-theorem coverage table](https://lizn-zn.github.io/VeriTile/proofs/coverage/)
  for configured models, slices, and remaining gaps) plus FlashAttention-1
  forward, online softmax, Welford, LayerNorm, log-sum-exp.
- **CI gates**: `.github/workflows/bench-audit.yml` runs
  `bench/audit_tritonbench_g.sh` — per-port elaboration
  (`bench/check_ports.sh`), the Python↔Lean faithfulness scans, the proof-gap
  manifest, and both trust audits (`#axiomsClean` over the library via
  `VeriTile.Meta.TrustReport`, and over the standalone bench corpus via
  `bench/audit_trust.sh`). `.github/workflows/artifact.yml` runs `lake build` +
  `scripts/check-artifact.sh` (no `sorry`, axiom whitelist, manifest schema,
  doc-drift checks). `.github/workflows/site.yml` builds and deploys the docs
  site to GitHub Pages on every push touching `site/`.

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
| One kernel matches an output spec | `ComputeCorrect.Realizes_without_Rounding` |
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
`scripts/prove.sh` automates this loop and uses the official comparator to judge
the targets selected with `--theorem` against the original task.
See [setup and usage](./scripts/README.md).
The artifact and bench check scripts also require comparator proof replay;
see [the shared verification gate](./scripts/README.md#shared-comparator-gate).

### 4. Register in the kernel manifest

Add a row to [`scripts/kernel-manifest.tsv`](./scripts/kernel-manifest.tsv)
so `scripts/check-artifact.sh` recognizes the theorem in CI. Schema and
naming conventions: [KernelManifest.md](./documents/KernelManifest.md),
[TheoremSurfaces.md](./documents/TheoremSurfaces.md).

## Minimal Example

Elementwise vector add against the `addSpec xs ys i = xs i + ys i` math
spec — see [`bench/examples/VectorAdd.lean`](./bench/examples/VectorAdd.lean).

## Refinement Example

Naive vs numerically-stable softmax (kernel pair refinement) — see
[`bench/examples/SoftmaxStableEquiv.lean`](./bench/examples/SoftmaxStableEquiv.lean).

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
bench/tritonbench_g/       TritonBench-G v1 ports (173 pairs; see completion_audit.md)
bench/examples/            Showcase proofs (SwiGLU rounding invariance, ...)
documents/                 Design notes, subset spec, surface guide
scripts/                   CI gate, kernel manifest, LLM proof wrapper
verso/                     Slide deck / overview
```

## Verification

- `lake build` — build the default `VeriTile` library target; standalone
  benchmarks/showcases and the GeLU/trust-report target are separate
- `lake build VeriTile VeriTileFull` — also build the full analysis and library trust report
- `lake env lean bench/examples/VectorAdd.lean` — quick example smoke check after building
- `scripts/check-artifact.sh` — `lake build` ∧ `no sorry` ∧ axiom
  whitelist ∧ kernel-manifest schema ∧ README/doc-term drift
- `bench/check_ports.sh` — per-port build of the TritonBench-G ports
  (also run inside `bench/audit_tritonbench_g.sh`, the bench-audit CI gate)
- `bench/audit_tritonbench_g.sh` — the full bench gate: the per-port build
  above ∧ faithfulness scans ∧ proof-gap manifest ∧ both trust audits
- `bench/audit_trust.sh` — trust gates and comparator replay for every standalone bench file

## Environment

- Lean 4 (`v4.29.0`) + Mathlib
- [Claude Code CLI](https://docs.claude.com/en/docs/claude-code) +
  [`lean4-skills`](https://github.com/lean4-skills/lean4-skills)
- For artifact/bench verification and proof automation: Python 3 and the official comparator, lean4export,
  and landrun on Linux with a systemd user service; see
  [installation instructions](./scripts/README.md#setup).

## Roadmap

Long-running project. Goal: bring real Triton kernels (forward, backward,
concurrency, production layouts / autograd) into Lean's proof scope with
minimal modification. Live roadmap:
[#1](https://github.com/Lizn-zn/VeriTile/issues/1). Architecture and
decision log: [PLAN.md](./PLAN.md).

## Team

VeriTile is a collaboration between Zenan Li, Kaiyu Yang, Ziran Yang,
Zhaoyu Li, Mike He, and Aarti Gupta.

## License

[MIT](./LICENSE) © 2026 Zenan Li.
