/-
VeriTile.Triton.Float.Refine

Rounding-invariant correctness surfaces (#447, the `ComputeRefine.*R` family).

Taxonomy (user decision on #447):
* `ComputeCorrect.*` — the erased/ideal pathway: correctness against an ℝ
  closed form, casts collapse. Unchanged; all existing bench theorems live
  there.
* `ComputeRefine.*R` — the rounding-model pathway: the kernel *refines* an
  `R`-annotated closed form, universally over all `RoundingModel`s. All
  structural information (how many rounding events, where) lives in where
  `R.round` occurs inside the spec term.

The two families meet in exactly one theorem: `ComputeRefine.Realizes.toRealizes_without_Rounding`
("refinement implies ideal correctness"), which instantiates `R := .triv` and
rides the Phase-A degeneration chain (`execR_triv`).

Note on the `R` suffix: the enclosing `ComputeRefine` namespace already hosts
the classic two-kernel refinement surfaces (`ComputeRefine.Refines_without_Rounding` compares
a kernel pair under the exact semantics); the rounding-parametric members
carry an `R` suffix to coexist with them.
-/

import VeriTile.Triton.Float.StepR
import VeriTile.Triton.Correctness

namespace VeriTile.Triton

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

/-- Every rounded value is representable (rounding is idempotent by definition
of `RoundingModel`). -/
theorem representable_round (R : RoundingModel)
    (dtype : FloatDType) (x : ℝ) :
    R.Representable dtype (R.round dtype x) :=
  R.round_idem dtype x

end RoundingModel

/-! ## Kernel-level correctness under a rounding model -/

namespace Kernel

/-- Postcondition-style correctness under a rounding model: `Kernel.Correct_without_Rounding`
with execution replaced by `execR R`. -/
def Correct (R : RoundingModel) (k : Kernel)
    (post : BlockState → BlockState → Prop) : Prop :=
  ∀ s s', execR R k s = some s' → post s s'

/-- Degeneration: at the trivial model, `Correct` *is* `Correct_without_Rounding`. -/
theorem correctR_triv_iff (k : Kernel) (post : BlockState → BlockState → Prop) :
    Correct .triv k post ↔ Correct_without_Rounding k post := by
  unfold Correct Correct_without_Rounding
  simp only [execR_triv]

/-- Two-kernel refinement under a rounding model: `Kernel.Refine` with both
executions replaced by `execR R`. -/
def RefineR (R : RoundingModel) (lhs rhs : Kernel)
    (rel : BlockState → BlockState → BlockState → Prop) : Prop :=
  ∀ s lhs' rhs',
    execR R lhs s = some lhs' →
    execR R rhs s = some rhs' →
    rel s lhs' rhs'

theorem refineR_triv_iff (lhs rhs : Kernel)
    (rel : BlockState → BlockState → BlockState → Prop) :
    RefineR .triv lhs rhs rel ↔ Refine lhs rhs rel := by
  unfold RefineR Refine
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

/-- `AlgorithmCorrect_without_Rounding` under a rounding model. -/
def AlgorithmCorrect (R : RoundingModel) (ck : ComputeKernel)
    (post : AlgSpec) : Prop :=
  match ck.toAlgorithm? with
  | Except.ok ak => Kernel.Correct R ak post
  | Except.error _ => False

theorem algorithmCorrectR_triv_iff (ck : ComputeKernel) (post : AlgSpec) :
    AlgorithmCorrect .triv ck post ↔ AlgorithmCorrect_without_Rounding ck post := by
  unfold AlgorithmCorrect AlgorithmCorrect_without_Rounding
  cases ck.toAlgorithm?
  · exact Iff.rfl
  · exact Kernel.correctR_triv_iff _ _

/-- `ComputeCorrect` under a rounding model (same `GapPolicy` plumbing). -/
def ComputeCorrectR (R : RoundingModel) (ck : ComputeKernel) (post : AlgSpec)
    (gap : GapPolicy := .ignore) : Prop :=
  GapPolicy.Holds gap ∧ AlgorithmCorrect R ck post

/-- One-run compute-facing correctness under a rounding model. -/
def ExecCorrectR (R : RoundingModel) (ck : ComputeKernel) (s : BlockState)
    (post : BlockState → Prop) : Prop :=
  ComputeCorrectR R ck (fun s0 s' => s0 = s → post s')

theorem execCorrectR_triv_iff (ck : ComputeKernel) (s : BlockState)
    (post : BlockState → Prop) :
    ExecCorrectR .triv ck s post ↔ ExecCorrect ck s post := by
  unfold ExecCorrectR ComputeCorrectR ExecCorrect ComputeCorrect ProjectedCorrect
  exact and_congr_right fun _ => algorithmCorrectR_triv_iff ck _

/-- Bridge for projected algorithm kernels using `ck.toAlgKernel` as the
canonical projection — the `R`-parametric mirror of
`computeCorrect_of_toAlgKernel`. -/
theorem computeCorrectR_of_toAlgKernel {R : RoundingModel} {ck : ComputeKernel}
    {post : AlgSpec}
    (h : ck.toAlgorithm? = Except.ok ck.toAlgKernel)
    (hc : Kernel.Correct R ck.toAlgKernel post) :
    ComputeCorrectR R ck post := by
  refine ⟨trivial, ?_⟩
  unfold AlgorithmCorrect
  rw [h]
  exact hc

/-- Unpack a fixed-`R` `ExecCorrectR` at one successful execution of the
projected kernel: the raw post-condition on the final state. Kernel-agnostic. -/
theorem ExecCorrectR.out {R : RoundingModel} {ck : ComputeKernel} {s : BlockState}
    {post : BlockState → Prop}
    (h : ExecCorrectR R ck s post)
    (hAlg : ck.toAlgorithm? = Except.ok ck.toAlgKernel)
    {s' : BlockState} (hExec : execR R ck.toAlgKernel s = some s') : post s' := by
  have h' := h.2
  unfold AlgorithmCorrect at h'
  rw [hAlg] at h'
  exact h' s s' hExec rfl

/-- `ProjectedRefine` under a rounding model. -/
def ProjectedRefineR (R : RoundingModel) (lhs rhs : ComputeKernel)
    (rel : BlockState → BlockState → BlockState → Prop) : Prop :=
  match lhs.toAlgorithm?, rhs.toAlgorithm? with
  | Except.ok lhsAlg, Except.ok rhsAlg => Kernel.RefineR R lhsAlg rhsAlg rel
  | _, _ => False

/-- `ComputeRefine` under a rounding model (same `GapPolicy` plumbing). -/
def ComputeRefineR (R : RoundingModel) (lhs rhs : ComputeKernel)
    (rel : BlockState → BlockState → BlockState → Prop)
    (gap : GapPolicy := .ignore) : Prop :=
  GapPolicy.Holds gap ∧ ProjectedRefineR R lhs rhs rel

/-- One-run refinement under a rounding model from a fixed initial state. -/
def ExecRefineR (R : RoundingModel) (lhs rhs : ComputeKernel) (s : BlockState)
    (rel : BlockState → BlockState → Prop) : Prop :=
  ComputeRefineR R lhs rhs (fun s0 lhs' rhs' => s0 = s → rel lhs' rhs')

theorem execRefineR_triv_iff (lhs rhs : ComputeKernel) (s : BlockState)
    (rel : BlockState → BlockState → Prop) :
    ExecRefineR .triv lhs rhs s rel ↔ ExecRefine lhs rhs s rel := by
  unfold ExecRefineR ComputeRefineR ProjectedRefineR
    ExecRefine ComputeRefine ProjectedRefine
  refine and_congr_right fun _ => ?_
  cases lhs.toAlgorithm? <;> cases rhs.toAlgorithm? <;>
    first
      | exact Iff.rfl
      | exact Kernel.refineR_triv_iff _ _ _

/-- Bridge for projected refinements using `toAlgKernel` on both sides — the
`R`-parametric mirror of `computeRefine_of_toAlgKernel`. -/
theorem computeRefineR_of_toAlgKernel {R : RoundingModel}
    {lhs rhs : ComputeKernel}
    {rel : BlockState → BlockState → BlockState → Prop}
    (hL : lhs.toAlgorithm? = Except.ok lhs.toAlgKernel)
    (hR : rhs.toAlgorithm? = Except.ok rhs.toAlgKernel)
    (h : Kernel.RefineR R lhs.toAlgKernel rhs.toAlgKernel rel) :
    ComputeRefineR R lhs rhs rel := by
  refine ⟨trivial, ?_⟩
  unfold ProjectedRefineR
  rw [hL, hR]
  exact h

end ComputeKernel

/-! ## The rounding-invariant realization surface -/

namespace ComputeRefine

/--
Rounding-invariant realization: for **every** rounding model `R`, executing
the kernel writes `expected R` at the mapped output cells.

This is the `ComputeRefine`-family analogue of `ComputeCorrect.Realizes_without_Rounding`,
with the spec upgraded from an ℝ closed form to a function of the model. A
kernel satisfying this exposes its complete rounding-event structure in the
shape of `expected`:
* `expected R i = core i` (no `R`) — the observed path is rounding-free;
* `expected R i = R.round dt (core i)` — exactly one final quantization;
* nested/iterated `R.round` — one event per occurrence.
-/
def Realizes {ι : Type} {α : Type} [ComputeCorrect.OutputReadable α]
    (kernel : ComputeKernel) (initialState : BlockState)
    (write : ComputeCorrect.WriteMap ι)
    (expected : RoundingModel → ι → α) : Prop :=
  ∀ R : RoundingModel,
    ComputeKernel.ExecCorrectR R kernel initialState (fun s' =>
      ∀ i : ι, match write i with
        | some addr => ComputeCorrect.OutputReadable.read s' addr = expected R i
        | none => True)

/-- `writeIf` unpacking for `Realizes` — the `R`-parametric mirror of
`ComputeCorrect.realizes_writeIf_iff`. -/
theorem realizes_writeIf_iff {ι : Type} {α : Type}
    [ComputeCorrect.OutputReadable α]
    (kernel : ComputeKernel) (initialState : BlockState)
    (mask : ι → Prop) [DecidablePred mask]
    (addr : ι → MemCellAddr) (expected : RoundingModel → ι → α) :
    Realizes kernel initialState
        (ComputeCorrect.WriteMap.writeIf mask addr) expected ↔
      ∀ R : RoundingModel,
        ComputeKernel.ExecCorrectR R kernel initialState (fun s' =>
          ∀ i : ι, mask i →
            ComputeCorrect.OutputReadable.read s' (addr i) = expected R i) := by
  unfold Realizes
  refine forall_congr' fun R => ?_
  unfold ComputeCorrect.WriteMap.writeIf ComputeKernel.ExecCorrectR
    ComputeKernel.ComputeCorrectR ComputeKernel.AlgorithmCorrect
    Kernel.Correct
  cases hAlg : kernel.toAlgorithm? with
  | error e => simp
  | ok ak =>
      simp only [and_congr_right_iff]
      intro _
      constructor
      · intro h s0 s' hExec hs0 i hi
        have hout := h s0 s' hExec hs0 i
        simpa [hi] using hout
      · intro h s0 s' hExec hs0 i
        by_cases hi : mask i
        · simpa [hi] using h s0 s' hExec hs0 i hi
        · simp [hi]

/-- **Refinement implies ideal correctness** — the single theorem connecting
the `ComputeRefine` family to the `ComputeCorrect` family. Instantiates the
universal statement at the trivial model, where the parametric semantics is
the exact semantics (Phase-A degeneration chain). -/
theorem Realizes.toRealizes_without_Rounding {ι : Type} {α : Type}
    [ComputeCorrect.OutputReadable α]
    {kernel : ComputeKernel} {initialState : BlockState}
    {write : ComputeCorrect.WriteMap ι}
    {expected : RoundingModel → ι → α}
    (h : Realizes kernel initialState write expected) :
    ComputeCorrect.Realizes_without_Rounding kernel initialState write (expected .triv) := by
  have := h .triv
  rw [ComputeKernel.execCorrectR_triv_iff] at this
  exact this

/--
Two-kernel *pointwise* refinement realization under a rounding model — the
`R`-parametric mirror of `ComputeRefine.RefinesAt_without_Rounding`. `relation` receives the
two kernels' output cells at declared addresses and can express the
event-ledger relation (each side equals its `R`-annotated term). The
canonical whole-memory refinement semantics is `ComputeRefine.Refines`.
-/
def RefinesAt (R : RoundingModel) {ι : Type} {α β : Type}
    [ComputeCorrect.OutputReadable α] [ComputeCorrect.OutputReadable β]
    (lhs rhs : ComputeKernel) (initialState : BlockState)
    (lhsWrite rhsWrite : ComputeCorrect.WriteMap ι)
    (relation : ι → α → β → Prop) : Prop :=
  ComputeKernel.ExecRefineR R lhs rhs initialState (fun lhs' rhs' =>
    ∀ i : ι, match lhsWrite i, rhsWrite i with
      | some lhsAddr, some rhsAddr =>
          relation i
            (ComputeCorrect.OutputReadable.read lhs' lhsAddr)
            (ComputeCorrect.OutputReadable.read rhs' rhsAddr)
      | _, _ => True)

/-- Degeneration: at the trivial model, `RefinesAt` *is* the classic
pointwise pair surface `ComputeRefine.RefinesAt_without_Rounding`. -/
theorem refinesAt_triv_iff {ι : Type} {α β : Type}
    [ComputeCorrect.OutputReadable α] [ComputeCorrect.OutputReadable β]
    (lhs rhs : ComputeKernel) (initialState : BlockState)
    (lhsWrite rhsWrite : ComputeCorrect.WriteMap ι)
    (relation : ι → α → β → Prop) :
    RefinesAt .triv lhs rhs initialState lhsWrite rhsWrite relation ↔
      RefinesAt_without_Rounding lhs rhs initialState lhsWrite rhsWrite relation := by
  unfold RefinesAt RefinesAt_without_Rounding
  exact ComputeKernel.execRefineR_triv_iff _ _ _ _

/-- `lhs` refines `rhs` under the rounding model `R`: from the same initial
state, the two kernels performed THE SAME WRITES — the final memories agree
at every cell outside the declared `scratch` regions. The `R`-parametric
mirror of the canonical `ComputeRefine.Refines_without_Rounding`. -/
def Refines (R : RoundingModel) (lhs rhs : ComputeKernel)
    (initialState : BlockState) (scratch : List RegionName := []) : Prop :=
  ComputeKernel.ExecRefineR R lhs rhs initialState (fun lhs' rhs' =>
    ∀ r, r ∉ scratch → ∀ o, lhs'.mem r o = rhs'.mem r o)

/-- Degeneration: at the trivial model, `Refines` *is* the canonical
whole-memory pair surface `ComputeRefine.Refines_without_Rounding`. -/
theorem refines_triv_iff (lhs rhs : ComputeKernel) (initialState : BlockState)
    (scratch : List RegionName) :
    Refines .triv lhs rhs initialState scratch ↔
      Refines_without_Rounding lhs rhs initialState scratch := by
  unfold Refines Refines_without_Rounding
  exact ComputeKernel.execRefineR_triv_iff _ _ _ _

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

/-- `RefinesAt` is monotone in the relation: any pointwise weakening of the
pair relation is again realized. -/
theorem RefinesAt.mono {R : RoundingModel} {ι : Type} {α β : Type}
    [ComputeCorrect.OutputReadable α] [ComputeCorrect.OutputReadable β]
    {lhs rhs : ComputeKernel} {initialState : BlockState}
    {lhsWrite rhsWrite : ComputeCorrect.WriteMap ι}
    {rel rel' : ι → α → β → Prop}
    (h : RefinesAt R lhs rhs initialState lhsWrite rhsWrite rel)
    (himp : ∀ i a b, rel i a b → rel' i a b) :
    RefinesAt R lhs rhs initialState lhsWrite rhsWrite rel' := by
  unfold RefinesAt ComputeKernel.ExecRefineR ComputeKernel.ComputeRefineR
    ComputeKernel.ProjectedRefineR at h ⊢
  obtain ⟨-, h⟩ := h
  refine ⟨trivial, ?_⟩
  cases hL : lhs.toAlgorithm? with
  | error e => rw [hL] at h; exact h
  | ok lAlg =>
      cases hR : rhs.toAlgorithm? with
      | error e => rw [hL, hR] at h; exact h
      | ok rAlg =>
          rw [hL, hR] at h
          intro s0 l r hl hr hs0 i
          have hi := h s0 l r hl hr hs0 i
          revert hi
          cases lhsWrite i <;> cases rhsWrite i <;> intro hi
          · trivial
          · trivial
          · trivial
          · exact himp _ _ _ hi

/-- A `RoundFree` realization yields the exact ℝ statement for every model,
not only the trivial one. -/
theorem Realizes.exact_of_roundFree {ι : Type} {α : Type}
    [ComputeCorrect.OutputReadable α]
    {kernel : ComputeKernel} {initialState : BlockState}
    {write : ComputeCorrect.WriteMap ι}
    {expected : RoundingModel → ι → α}
    (h : Realizes kernel initialState write expected)
    (hfree : RoundFree expected) (R : RoundingModel) :
    ComputeKernel.ExecCorrectR R kernel initialState (fun s' =>
      ∀ i : ι, match write i with
        | some addr =>
            ComputeCorrect.OutputReadable.read s' addr = expected .triv i
        | none => True) := by
  have := h R
  rwa [hfree R] at this

/-- Unpack a `Realizes` statement at one model and one successful execution of
the projected kernel: the raw per-lane output clause. Kernel-agnostic — usable
by any `Realizes` consumer that needs the per-write conclusion at a concrete
execution. -/
theorem Realizes.out {ι : Type} {α : Type}
    [ComputeCorrect.OutputReadable α]
    {k : ComputeKernel} {st : BlockState}
    {write : ComputeCorrect.WriteMap ι} {expected : RoundingModel → ι → α}
    (h : Realizes k st write expected) (R : RoundingModel)
    (hAlg : k.toAlgorithm? = Except.ok k.toAlgKernel)
    {s' : BlockState}
    (hExec : execR R k.toAlgKernel st = some s') :
    ∀ i : ι, match write i with
      | some addr => ComputeCorrect.OutputReadable.read s' addr = expected R i
      | none => True := by
  have h' := (h R).2
  unfold ComputeKernel.AlgorithmCorrect at h'
  rw [hAlg] at h'
  exact h' st s' hExec rfl

/-- Unpack a `Refines` statement at one successful execution pair: the raw
outside-scratch memory-agreement clause. Kernel-agnostic. -/
theorem Refines.memAgree {R : RoundingModel} {lhs rhs : ComputeKernel}
    {st : BlockState} {scratch : List RegionName}
    (h : Refines R lhs rhs st scratch)
    (hLAlg : lhs.toAlgorithm? = Except.ok lhs.toAlgKernel)
    (hRAlg : rhs.toAlgorithm? = Except.ok rhs.toAlgKernel)
    {l r : BlockState}
    (hEL : execR R lhs.toAlgKernel st = some l)
    (hER : execR R rhs.toAlgKernel st = some r) :
    ∀ reg ∉ scratch, ∀ o : Nat, l.mem reg o = r.mem reg o := by
  unfold Refines ComputeKernel.ExecRefineR
    ComputeKernel.ComputeRefineR ComputeKernel.ProjectedRefineR at h
  rw [hLAlg, hRAlg] at h
  exact h.2 st l r hEL hER rfl

end ComputeRefine

end VeriTile.Triton
