/-
VeriTile.Float.Refine

Rounding-invariant correctness surfaces (#447, the `ComputeRefine.*R` family).

Taxonomy (user decision on #447):
* `ComputeCorrect.*` — the erased/ideal pathway: correctness against an ℝ
  closed form, casts collapse. Unchanged; all existing bench theorems live
  there.
* `ComputeRefine.*R` — the rounding-model pathway: the kernel *refines* an
  `R`-annotated closed form, universally over all `RoundingModel`s. All
  structural information (how many rounding events, where) lives in where
  `R.round` occurs inside the spec term.

The two families meet in exactly one theorem: `ComputeRefine.RealizesR.toRealizes`
("refinement implies ideal correctness"), which instantiates `R := .triv` and
rides the Phase-A degeneration chain (`execR_triv`).

Note on the `R` suffix: the enclosing `ComputeRefine` namespace already hosts
the classic two-kernel refinement surfaces (`ComputeRefine.Realizes` compares
a kernel pair under the exact semantics); the rounding-parametric members
carry an `R` suffix to coexist with them.
-/

import VeriTile.Float.StepR
import VeriTile.Float.Correctness

namespace VeriTile

/-! ## Fixed points of a rounding model -/

namespace RoundingModel

/-- `x` is representable for `dtype` under `R` iff rounding fixes it. This is
the *derived* notion of representability — no grids, no mantissas. -/
def Representable (R : RoundingModel) (dtype : FloatDType) (x : ℝ) : Prop :=
  R.round dtype x = x

@[simp] theorem representable_triv (dtype : FloatDType) (x : ℝ) :
    triv.Representable dtype x := rfl

@[simp] theorem representable_real (R : RoundingModel) (x : ℝ) :
    R.Representable .real x := R.round_real_apply x

/-- Under an idempotent model, every rounded value is representable. -/
theorem representable_round (R : RoundingModel) [IdemRounding R]
    (dtype : FloatDType) (x : ℝ) :
    R.Representable dtype (R.round dtype x) :=
  IdemRounding.idem dtype x

end RoundingModel

/-! ## Kernel-level correctness under a rounding model -/

namespace Kernel

/-- Postcondition-style correctness under a rounding model: `Kernel.Correct`
with execution replaced by `execR R`. -/
def CorrectR (R : RoundingModel) (k : Kernel)
    (post : BlockState → BlockState → Prop) : Prop :=
  ∀ s s', execR R k s = some s' → post s s'

/-- Degeneration: at the trivial model, `CorrectR` *is* `Correct`. -/
theorem correctR_triv_iff (k : Kernel) (post : BlockState → BlockState → Prop) :
    CorrectR .triv k post ↔ Correct k post := by
  unfold CorrectR Correct
  simp only [execR_triv]

end Kernel

/-! ## Compute-kernel surfaces under a rounding model -/

namespace ComputeKernel

/-- Compute-kernel execution under a rounding model. The algorithm projection
`toAlgorithm?` preserves `castFloat` nodes (it does not erase dtypes), so the
projected kernel exposes every cast event to `execR`. -/
noncomputable def evalR (R : RoundingModel) (ck : ComputeKernel)
    (s : BlockState) : Option BlockState :=
  match ck.toAlgorithm? with
  | Except.ok ak => execR R ak s
  | Except.error _ => none

@[simp] theorem evalR_triv (ck : ComputeKernel) (s : BlockState) :
    evalR .triv ck s = eval ck s := by
  unfold evalR eval
  cases ck.toAlgorithm? <;> simp [execR_triv]

/-- `AlgorithmCorrect` under a rounding model. -/
def AlgorithmCorrectR (R : RoundingModel) (ck : ComputeKernel)
    (post : AlgSpec) : Prop :=
  match ck.toAlgorithm? with
  | Except.ok ak => Kernel.CorrectR R ak post
  | Except.error _ => False

theorem algorithmCorrectR_triv_iff (ck : ComputeKernel) (post : AlgSpec) :
    AlgorithmCorrectR .triv ck post ↔ AlgorithmCorrect ck post := by
  unfold AlgorithmCorrectR AlgorithmCorrect
  cases ck.toAlgorithm?
  · exact Iff.rfl
  · exact Kernel.correctR_triv_iff _ _

/-- `ComputeCorrect` under a rounding model (same `GapPolicy` plumbing). -/
def ComputeCorrectR (R : RoundingModel) (ck : ComputeKernel) (post : AlgSpec)
    (gap : GapPolicy := .ignore) : Prop :=
  GapPolicy.Holds gap ∧ AlgorithmCorrectR R ck post

/-- One-run compute-facing correctness under a rounding model. -/
def ExecCorrectR (R : RoundingModel) (ck : ComputeKernel) (s : BlockState)
    (post : BlockState → Prop) : Prop :=
  ComputeCorrectR R ck (fun s0 s' => s0 = s → post s')

theorem execCorrectR_triv_iff (ck : ComputeKernel) (s : BlockState)
    (post : BlockState → Prop) :
    ExecCorrectR .triv ck s post ↔ ExecCorrect ck s post := by
  unfold ExecCorrectR ComputeCorrectR ExecCorrect ComputeCorrect ProjectedCorrect
  exact and_congr_right fun _ => algorithmCorrectR_triv_iff ck _

end ComputeKernel

/-! ## The rounding-invariant realization surface -/

namespace ComputeRefine

/--
Rounding-invariant realization: for **every** rounding model `R`, executing
the kernel writes `expected R` at the mapped output cells.

This is the `ComputeRefine`-family analogue of `ComputeCorrect.Realizes`,
with the spec upgraded from an ℝ closed form to a function of the model. A
kernel satisfying this exposes its complete rounding-event structure in the
shape of `expected`:
* `expected R i = core i` (no `R`) — the observed path is rounding-free;
* `expected R i = R.round dt (core i)` — exactly one final quantization;
* nested/iterated `R.round` — one event per occurrence.
-/
def RealizesR {ι : Type} {α : Type} [ComputeCorrect.OutputReadable α]
    (kernel : ComputeKernel) (initialState : BlockState)
    (write : ComputeCorrect.WriteMap ι)
    (expected : RoundingModel → ι → α) : Prop :=
  ∀ R : RoundingModel,
    ComputeKernel.ExecCorrectR R kernel initialState (fun s' =>
      ∀ i : ι, match write i with
        | some addr => ComputeCorrect.OutputReadable.read s' addr = expected R i
        | none => True)

/-- **Refinement implies ideal correctness** — the single theorem connecting
the `ComputeRefine` family to the `ComputeCorrect` family. Instantiates the
universal statement at the trivial model, where the parametric semantics is
the exact semantics (Phase-A degeneration chain). -/
theorem RealizesR.toRealizes {ι : Type} {α : Type}
    [ComputeCorrect.OutputReadable α]
    {kernel : ComputeKernel} {initialState : BlockState}
    {write : ComputeCorrect.WriteMap ι}
    {expected : RoundingModel → ι → α}
    (h : RealizesR kernel initialState write expected) :
    ComputeCorrect.Realizes kernel initialState write (expected .triv) := by
  have := h .triv
  rw [ComputeKernel.execCorrectR_triv_iff] at this
  exact this

/-! ### Named invariant shapes -/

/-- The spec is independent of the rounding model: the observed dataflow
contains **no** rounding events, so the kernel's outputs equal the ℝ closed
form exactly, for every rounding behavior. This is the gating-GEMM shape. -/
def RoundFree {ι : Type} {α : Type}
    (expected : RoundingModel → ι → α) : Prop :=
  ∀ R : RoundingModel, expected R = expected .triv

/-- The spec is exactly one final quantization of an ℝ core: the strongest
statement a kernel with a narrow output channel can satisfy. This is the
fused-SwiGLU shape. -/
def SingleRounding {ι : Type}
    (expected : RoundingModel → ι → ℝ) (dtype : FloatDType)
    (core : ι → ℝ) : Prop :=
  ∀ (R : RoundingModel) (i : ι), expected R i = R.round dtype (core i)

/-- A `RoundFree` realization yields the exact ℝ statement for every model,
not only the trivial one. -/
theorem RealizesR.exact_of_roundFree {ι : Type} {α : Type}
    [ComputeCorrect.OutputReadable α]
    {kernel : ComputeKernel} {initialState : BlockState}
    {write : ComputeCorrect.WriteMap ι}
    {expected : RoundingModel → ι → α}
    (h : RealizesR kernel initialState write expected)
    (hfree : RoundFree expected) (R : RoundingModel) :
    ComputeKernel.ExecCorrectR R kernel initialState (fun s' =>
      ∀ i : ι, match write i with
        | some addr =>
            ComputeCorrect.OutputReadable.read s' addr = expected .triv i
        | none => True) := by
  have := h R
  rwa [hfree R] at this

end ComputeRefine

end VeriTile
