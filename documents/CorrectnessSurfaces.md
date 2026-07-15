# Correctness Surfaces

This document explains which public theorem surface to use when proving
properties of `ComputeKernel`s. Surfaces live in
[`VeriTile.Triton.Correctness`](../VeriTile.Triton.Correctness.lean).

## The rounding default

Read this first — it fixes what every surface name below means. The **unqualified**
surface names (`Realizes`, `Refines`, `RefinesAt`, `Correct`) are the
**rounding-model** surfaces: they take a `RoundingModel R` and execute under the
R-threaded semantics `execR`. The **exact-ℝ idealization** — the older "everything
is real arithmetic, no quantization" reading — is the explicitly-qualified
`*_without_Rounding` surface (`Realizes_without_Rounding`,
`Refines_without_Rounding`, `RefinesAt_without_Rounding`, `Correct_without_Rounding`).

Rule of thumb: the unqualified name is what a kernel *actually* does on hardware
(rounded); `*_without_Rounding` is the mathematician's idealization you fall back
to when the rounding is irrelevant to the claim. The exact surface **degenerates
out of** the rounding one at the trivial model `.triv` (where `execR` collapses
onto `exec`), never the other way around. Most of the 141 ported kernels are
proven on `*_without_Rounding` (their outputs are exact-ℝ specs); the showcase
pairs under `bench/examples/` that carry a bf16 boundary store land on the
rounding surface `Refines R`.

## Quick-Pick Table

| Goal | Use |
|---|---|
| One kernel realizes an output spec (rounded) | `ComputeCorrect.Realizes` |
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
the launcher exists, grid theorems will move to `ComputeCorrect.Realizes` with
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
  outputs). `ComputeCorrect.Realizes` is the rounding form;
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
ComputeRefine.RefinesAt lhs rhs s lhsWrite rhsWrite relation
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
`RoundingModel R` (`round : FloatDType → ℝ → ℝ`, with idempotence a defining
field) and executes under the R-threaded semantics `execR`. They live in
[`VeriTile.Triton.Float.Refine`](../VeriTile/Triton/Float/Refine.lean):

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
`execR` collapses onto `exec`): the bridge `ComputeRefine.Realizes.toRealizes`
("rounding claim implies ideal correctness") turns any `Realizes` into an
ordinary `ComputeCorrect.Realizes_without_Rounding` for `expected .triv`.
Degeneration lemmas `refines_triv_iff` / `refinesAt_triv_iff` recover
`Refines_without_Rounding` / `RefinesAt_without_Rounding` the same way. The
gold-standard walkthrough for the ∀R compositional pattern is
[`bench/examples/FusedSwigluEquiv.lean`](../bench/examples/FusedSwigluEquiv.lean); the
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
[`EraseDType.md`](./EraseDType.md) for the compute-to-algorithm projection
policy.

## Naming

Recommended theorem names:

- `<kernel>_compute_correct` for ordinary single-kernel correctness.
- `<kernel>_correct_view` for public manifest-compatible view surfaces.
- `<rewrite>_refinement_view` for public two-kernel refinement surfaces.

Execution-only helper lemmas may use `_exec_view` and direct `exec`
equalities. The public theorem should wrap those helpers in
`ComputeCorrect.*` or `ComputeRefine.*`. Naming details:
[`TheoremSurfaces.md`](./TheoremSurfaces.md).
