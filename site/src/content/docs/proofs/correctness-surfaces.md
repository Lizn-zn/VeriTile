---
title: "Correctness Surfaces"
---

This document explains which public theorem surface to use when proving
properties of `ComputeKernel`s. Exact surfaces live in
[`VeriTile.Triton.Correctness`](https://github.com/Lizn-zn/VeriTile/blob/main/VeriTile/Triton/Correctness.lean); rounding
surfaces live in [`Float.Refine`](https://github.com/Lizn-zn/VeriTile/blob/main/VeriTile/Triton/Float/Refine.lean).

## Rounding models and exact surfaces

The interfaces are defined in `VeriTile/Triton/Correctness.lean` (exact
semantics) and `VeriTile/Triton/Float/Refine.lean` (rounding models):

- `ComputeCorrect.Realizes_without_Rounding kernel s write expected` observes
  outputs under exact semantics, with `expected : ι → α`.
- `ComputeRefine.Realizes kernel s write expected` quantifies over all
  `RoundingModel`s, with `expected : RoundingModel → ι → α`.
- `ComputeRefine.Refines R lhs rhs s scratch` and `RefinesAt R ...` compare two
  kernels under a supplied model `R`. Their exact versions have the
  `_without_Rounding` suffix.
- `ComputeCorrect.Post`, `General`, and the `Output*` wrappers retain exact semantics.

A `RoundingModel` requires identity on the real channel (`round_real`) and
idempotence (`round_idem`). Monotonicity, oddness, and grid nesting are optional
hypotheses. The model records rounding at explicit float casts and stores;
it does not model IEEE-754 rounding at every arithmetic operation or establish
numerical error bounds. At `R := .triv`, `execR` degenerates to `exec`; the
bridge lemmas are described below.

## KernelIO `⊨` — the per-kernel headline surface

The surfaces below (`Realizes`, `Refines`, …) are the *library* vocabulary. The
surface that the bench corpus actually states its headlines on is the **KernelIO
`⊨` triple**, defined in
[`VeriTile/Triton/Memory/KernelSpec.lean`](https://github.com/Lizn-zn/VeriTile/blob/main/VeriTile/Triton/Memory/KernelSpec.lean). The signature variants cover different input/output arities, masks,
indices, and streaming windows. Consult the
[proof-gap manifest](https://github.com/Lizn-zn/VeriTile/blob/main/bench/tritonbench_g/proof_gap_manifest.tsv) for
per-port coverage.

### What a `KernelIO` is

A `KernelIO*` record is a kernel's **IO signature**: which regions it reads and
writes, where program `pid`'s window starts in each of them (`read`/`write`),
which lanes are active (`readMask`/`writeMask`), the tile length `B`, and — on
the rounding skins — the output element type `outDType`. The many `structure`s
("skins") differ only in the *shape* of that signature: how many inputs and
outputs, how many program axes, whether metadata scalars or a gather index sit
between the pointer and the data.

### What `io ⊨ f` says

```lean
open scoped VeriTile.Triton.KernelIO₂ in
specification my_kernel_correctness … :
    myIO … ⊨ fun xs ys j => xs j + ys j
```

Unfolded, that is one Hoare triple over **flat pointer memory**: for every
disjoint flat placement of the declared buffers, every program id whose windows
are in bounds, and every launch state whose input windows hold `xs`/`ys`, the
*translated pointer kernel* terminates, every declared output cell holds
`f xs ys j`, and **every other flat cell is unchanged** (the frame).

That last conjunct is the part a hand-rolled `∃ sF, exec … = some sF ∧ …`
headline usually omits, which is why it is not merely a different spelling —
see `proof_blockers.md`'s "Correctness-Surface Blockers" for the ports that
still owe it.

### The notations

| Notation | Meaning |
|---|---|
| `io ⊨ f` | exact-ℝ: `exec`, no rounding |
| `io ⊨[R] f` | rounding: `execR R`, output rounded at the signature's `outDType` |
| `io ⊨[R, dtype] f` | same, with `outDType` written out — use this whenever a file states more than one output channel, since the 3-hole form prints an `.fp16` face and a `.real` face identically |
| `io₁ ≡[R] io₂` | kernel-vs-kernel refinement, the `⊨` counterpart of `Refines` |

`io ⊨[R] f` uses the supplied model `R`; universal quantification must be
explicit in the theorem. A theorem valid for every `R` can be instantiated at
`.triv`. Boundary quantization has narrowing content when the output dtype is
narrow; the `.real` channel satisfies `R.round .real = id`.

### Proving one

Each skin ships an assembly lemma `<Skin>.Implements.intro` (and
`.ImplementsR.intro`) that reduces `⊨` to three per-kernel obligations and does
the flat-memory transport once:

1. `FlattenOk` — the kernel's ops are in the flat-memory bridge fragment;
2. `TraceSafe` — the per-execution safety walk, given the window-in-bounds contract;
3. `hrun` — the **region-model** Hoare triple: termination, output values, frame.

Only (3) is mathematical content; (1) and (2) are mechanical walks.

## Quick-Pick Table

| Goal | Use |
|---|---|
| **A bench-port / showcase headline** | a `KernelIO` skin's `⊨` (exact) or `⊨[R]` (rounded) — see the section above |
| … kernel-vs-kernel, on the same surface | `io₁ ≡[R] io₂` |
| One kernel realizes an output spec (rounded) | `ComputeRefine.Realizes` |
| … the exact-ℝ idealization of that spec | `ComputeCorrect.Realizes_without_Rounding` |
| One kernel refines another, writes-equality (rounded) | `ComputeRefine.Refines` |
| … the exact-ℝ idealization | `ComputeRefine.Refines_without_Rounding` |
| Two kernels agree pointwise per declared address | `ComputeRefine.RefinesAt` (exact: `RefinesAt_without_Rounding`) |
| Whole-grid launch (every program writes correctly) | `Kernel.ForAllProgramsSome` (temporary; see Grid section) |
| Value/index pair on every lane | `ComputeCorrect.OutputPair` |
| Value/index pair on active lanes only | `ComputeCorrect.OutputPairWhere` |
| Two kernels produce equal value/index pairs | `ComputeRefine.OutputPairEq` / `OutputPairEqWhere` |
| Custom postcondition over the final state | `ComputeCorrect.Post` / `ComputeRefine.Post` |
| Relation over arbitrary initial states (rare) | `ComputeCorrect.General` / `ComputeRefine.General` |

The lower-level `ComputeKernel.ComputeCorrect` and
`ComputeKernel.ComputeRefine` definitions remain the implementation layer.
New example theorem statements should normally not expose those names directly.

## Execution, termination, and frame

`Realizes_without_Rounding` and `ComputeRefine.Realizes` constrain outputs of
successful executions; they do not by themselves establish success or
termination. `Refines` likewise compares the results when both executions
succeed. `RealizesFrame_without_Rounding` additionally preserves memory outside
the write map. For a bench headline that bundles termination, outputs, and
frame, prefer an appropriate `KernelIO` `⊨` / `⊨[R]` signature and inspect its
specific preconditions.

## Output Write Maps

The most general output surface separates two maps over the same logical output
index:

```lean
abbrev ComputeCorrect.WriteMap (ι : Type) := ι → Option MemCellAddr

ComputeCorrect.Realizes_without_Rounding
  (kernel := k)
  (initialState := s)
  (write := write)
  (expected := expected)
-- post: ∀ i, match write i with
--   | some addr => read final addr = expected i
--   | none => True
```

`Realizes_without_Rounding` is overloaded by the expected value type. `ℝ` specs use
`BlockState.readMem`, `Nat` specs use `readMemValue .nat`, and `MemCell` specs
use exact algorithm-layer cell equality. Use it when the spec is naturally a
write-indexed written-cell contract or spans multiple regions. For common
scalar/tensor readback theorems, the ergonomic wrappers (`OutputScalar`,
`OutputArray`, `OutputNatScalar`) remain available.

Masked stores should usually be stated with `WriteMap.writeIf`:

```lean
write := ComputeCorrect.WriteMap.writeIf
  (fun i : Fin BLOCK_SIZE => base + i.val < N)
  (fun i => (out, base + i.val))
```

Then the theorem states what active lanes wrote. Preservation of non-written
addresses belongs to frame/preserve theorems.

The standard proof step is:

```lean
rw [ComputeCorrect.realizes_writeIf_iff]
```

which turns the output obligation into `∀ i, mask i → read final (addr i) =
expected i`.

Whole-grid launch theorems should first expose a final-state execution surface;
once the kernel is represented as a final-state producing `ComputeKernel`
surface, the same `Realizes` form should describe its outputs. VeriTile does
not keep a separate `GridOutputAt` user surface; grid execution is an execution
concern, not a different output shape.

### Grid launches today

VeriTile does not yet have a whole-grid `launchExec : ComputeKernel → Grid →
BlockState → Option BlockState`. Until it lands, grid theorems are stated in
the per-program-local form `Kernel.ForAllProgramsSome`, which says: for every
typed grid index, running the kernel from `s.withGridIndex idx` produces a
state where the per-`idx` postcondition holds. See
`logsumexp_fwd_kernel_grid_blockLSE_correct` for the canonical example. Once
the launcher exists, grid theorems can use `ComputeCorrect.Realizes_without_Rounding` with
the launch as an extra parameter; the user-facing theorem shape stays the same.

## Single-Kernel Correctness

Single-kernel correctness checks one kernel against a mathematical or
algorithmic specification.

### Scalar outputs

```lean
ComputeCorrect.OutputScalar k s out offset expected
-- post: s'.readMem out offset = expected

ComputeCorrect.OutputNatScalar k s out offset expected
-- post: s'.readMemValue .nat out offset = expected
```

Use these for scalar reductions such as a final `max`, `sum`, or
`argmax`-style integer index.

### Tensor outputs

```lean
ComputeCorrect.OutputTile k s view expected
-- post: ∀ idx, TensorView.observe (some s') view idx = some (expected idx)

ComputeCorrect.OutputArray k s view expected   -- 1D specialization
-- post: same, with `expected : Fin n → ℝ`
```

`OutputArray` is the 1D `n`-shape specialization of `OutputTile` and is the
preferred surface when the spec is naturally a `Fin n → ℝ` function. Both are
thin wrappers over `Realizes_without_Rounding` through `WriteMap.ofTensorView`; they exist to
keep tensor-view theorem statements readable.

### Value/index pair outputs

```lean
ComputeCorrect.OutputPair k s valueRegion indexRegion offset
  expectedValue expectedIndex
-- post: every lane satisfies the value spec and the typed Nat index spec

ComputeCorrect.OutputPairWhere k s valueRegion indexRegion offset
  active expectedValue expectedIndex
-- post: same, but only on lanes where `active i` holds
```

Use these for kernels with paired outputs such as
`tl.max(..., return_indices=True)`. `OutputPairWhere` is the right choice
when masking restricts which lanes participate.

These remain dedicated definitions rather than `Realizes_without_Rounding` wrappers because the
two channels have different readback carriers (`ℝ` for value, `Nat` for index)
and the `Realizes_without_Rounding` typeclass dispatches on a single carrier per call. A future
revision could introduce a per-lane carrier typeclass to subsume them; not
worth doing until a second heterogeneous-output kernel appears.

## Kernel Refinement

### `Realizes` vs `Refines` — the naming scheme

VeriTile splits the two verification questions by name (each name is the
rounding surface; append `_without_Rounding` for the exact-ℝ idealization):

- **`Realizes`** — *a kernel realizes a spec* (one kernel vs expected
  outputs). `ComputeRefine.Realizes` is the rounding form;
  `ComputeCorrect.Realizes_without_Rounding` is the exact single-kernel
  workhorse that most ported kernels use.
- **`Refines`** — *a kernel refines another kernel* (two kernels compared to
  each other). `ComputeRefine.Refines` (writes-equality) and
  `ComputeRefine.RefinesAt` (pointwise per-address relation), with their exact
  mirrors `Refines_without_Rounding` / `RefinesAt_without_Rounding`.

### Writes-equality: `ComputeRefine.Refines`

The canonical two-kernel surface is `ComputeRefine.Refines`. Under a rounding
model `R` it runs both kernels from the same initial state through `execR R`
and asserts that they performed **the same writes** — the two final memories
agree at every cell outside a declared list of `scratch` regions:

```lean
ComputeRefine.Refines R lhs rhs s scratch
-- := ExecRefineR R lhs rhs s (fun l r =>
--      ∀ region ∉ scratch, ∀ offset, l.mem region offset = r.mem region offset)
```

Same write locations, same written values, one equation — for the given `R`.
`scratch` names the regions the two kernels are *allowed* to disagree on — a
pipeline's intermediate tensors (e.g. `[S]` for a fused-vs-unfused SwiGLU pair,
or the `zReg`/`siluReg` temporaries in `FusedSiLU`). Pass `[]` when the two
kernels must agree on all of memory. This is the surface every
`*_refinement_view` theorem in `bench/examples/` lands on; the exact-ℝ variant
`ComputeRefine.Refines_without_Rounding` (no `R`, running under `exec`) is the
idealization used by `bench/examples/FusedSiLUEquiv.lean`.

### Pointwise: `ComputeRefine.RefinesAt`

When the two kernels write to **different** target cells, or the comparison is
a non-equality relation, use the pointwise form. It relates the two kernels'
outputs through two independent `WriteMap`s:

```lean
ComputeRefine.RefinesAt R lhs rhs s lhsWrite rhsWrite relation
-- post: ∀ i, match lhsWrite i, rhsWrite i with
--   | some la, some ra => relation i (read lhs' la) (read rhs' ra)
--   | _, _ => True
```

The carrier types of the two reads are inferred independently, so `RefinesAt`
covers heterogeneous-layout comparisons and proof middleware. Use the same
write map on both sides for ordinary same-buffer equivalence. `Refines` is the
whole-memory form to reach for first; `RefinesAt` is the escape hatch when you
genuinely need per-side values.

### Rounding-model surfaces (narrow float, #447)

The unqualified surfaces are the rounding surfaces: each is parametric over a
`R : RoundingModel` (`round : FloatDType → ℝ → ℝ`, with idempotence a defining
field) and executes under the R-threaded semantics `execR`. They live in
[`VeriTile.Triton.Float.Refine`](https://github.com/Lizn-zn/VeriTile/blob/main/VeriTile/Triton/Float/Refine.lean):

- `ComputeRefine.Realizes kernel s write expected` — single kernel vs an
  `R`-annotated spec `expected : RoundingModel → ι → α`. The spec's shape
  *is* the rounding-event ledger: no `R.round` means the observed path is
  rounding-free; one `R.round` means one final quantization; nested
  `R.round`s count one event each.
- `ComputeRefine.Refines R lhs rhs s scratch` — the writes-equality pair
  surface under `R`.
- `ComputeRefine.RefinesAt R lhs rhs s lhsWrite rhsWrite relation` — the
  pointwise pair surface under `R`.

The exact-ℝ idealizations are the `*_without_Rounding` mirrors, and they
**degenerate out of** the rounding surfaces at the trivial model `.triv` (where
`execR` collapses onto `exec`): the bridge `ComputeRefine.Realizes.toRealizes_without_Rounding`
("rounding claim implies ideal correctness") turns any `Realizes` into an
ordinary `ComputeCorrect.Realizes_without_Rounding` for `expected .triv`.
Degeneration lemmas `refines_triv_iff` / `refinesAt_triv_iff` recover
`Refines_without_Rounding` / `RefinesAt_without_Rounding` the same way. The
gold-standard walkthrough for the ∀R compositional pattern is
[`bench/examples/FusedSwigluEquiv.lean`](https://github.com/Lizn-zn/VeriTile/blob/main/bench/examples/FusedSwigluEquiv.lean); the
boundary-rounding showcases (LogSumExp, softmax, Welford, LayerNorm, FusedSiLU)
land on `Refines R` for a fixed `R`.

### Equality helpers

For ordinary optimization proofs that read back the same buffer on both sides,
the equality helpers below are thin wrappers over `ExecRefine` and remain the
most readable choice.

### Scalar equality

```lean
ComputeRefine.OutputScalarEq lhs rhs s lhsOut lhsOffset rhsOut rhsOffset
-- post: lhs'.readMem lhsOut lhsOffset = rhs'.readMem rhsOut rhsOffset

ComputeRefine.OutputNatScalarEq lhs rhs s lhsOut lhsOffset rhsOut rhsOffset
-- post: lhs'.readMemValue .nat ... = rhs'.readMemValue .nat ...
```

### Tensor equality

```lean
ComputeRefine.OutputTileEq lhs rhs s lhsView rhsView
-- post: ∀ idx,
--   TensorView.observe (some lhs') lhsView idx
--     = TensorView.observe (some rhs') rhsView idx

ComputeRefine.OutputArrayEq lhs rhs s lhsView rhsView   -- 1D specialization
```

### Value/index pair equality

```lean
ComputeRefine.OutputPairEq lhs rhs s
  lhsValueRegion lhsIndexRegion rhsValueRegion rhsIndexRegion offset

ComputeRefine.OutputPairEqWhere lhs rhs s ... offset active
```

Use the `Where` variant when the equality only needs to hold on active lanes.

## Escape Hatches

`Post` fixes the initial state but accepts any postcondition over the
final state:

```lean
ComputeCorrect.Post k s (fun s' => ...)
ComputeRefine.Post lhs rhs s (fun lhs' rhs' => ...)
```

`General` quantifies over the initial state itself, taking a relation
over `(s0, s')`:

```lean
ComputeCorrect.General k (fun s0 s' => ...)
ComputeRefine.General lhs rhs (fun s0 lhs' rhs' => ...)
```

`General` should be reserved for theorems that genuinely need to be
parametric in the initial state (rare). Most proofs fix `s` and use
`Post`, or pick an `Output*` helper.

## Gap Policy

The `ComputeCorrect.*` / `ComputeRefine.*` helpers use the default
`GapPolicy.ignore`, matching current Real-first algorithm proofs. If a
theorem needs to record an externally checked compute-to-algorithm gap,
state the underlying `ComputeKernel.ComputeCorrect` /
`ComputeKernel.ComputeRefine` directly with:

```lean
gap := .require contract
```

or add a dedicated helper for that theorem family. See
[`EraseDType.md`](/VeriTile/architecture/erase-dtype/) for the compute-to-algorithm projection
policy.

## Naming

Recommended theorem names:

- `<kernel>_compute_correct` for ordinary single-kernel correctness.
- `<kernel>_correct_view` for public manifest-compatible view surfaces.
- `<rewrite>_refinement_view` for public two-kernel refinement surfaces.

Execution-only helper lemmas may use `_exec_view` and direct `exec`
equalities. The public theorem should wrap those helpers in
`ComputeCorrect.*` or `ComputeRefine.*`. Naming details:
[`TheoremSurfaces.md`](/VeriTile/proofs/theorem-surfaces/).
