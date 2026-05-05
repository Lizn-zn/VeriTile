/-
VeriTile.Triton.Float.Correctness

Float-facing correctness bridge for the current real-valued Triton semantics.
-/

import VeriTile.Triton.Float.StateErasure
import VeriTile.Triton.Memory
import VeriTile.Triton.ExternalCheck

namespace VeriTile.Triton

namespace Kernel

/-! ## Correctness layers -/

/-- A generic postcondition-style correctness predicate for executed kernels. -/
def Correct (k : Kernel) (post : BlockState → BlockState → Prop) : Prop :=
  ∀ s s', exec k s = some s' → post s s'

/--
A generic two-kernel refinement predicate.

`rel init lhsFinal rhsFinal` states what it means for the left kernel to refine
the right kernel from the same initial state. Many examples instantiate `rel`
with equality of selected observations rather than full-state equality.
-/
def Refine
    (lhs rhs : Kernel)
    (rel : BlockState → BlockState → BlockState → Prop) : Prop :=
  ∀ s lhs' rhs',
    exec lhs s = some lhs' →
    exec rhs s = some rhs' →
    rel s lhs' rhs'

/--
Algorithmic correctness for kernels that carry explicit `.fp32` / `.fp16` /
`.bf16` annotations in the current real-valued semantics.

The theorem is stated on the dtype-annotated kernel `k`, but the proof runs the
erased Real kernel `k.eraseDType`. This is the formal Lean proof layer for
float-facing kernels: state float, prove Real.
-/
def AlgorithmCorrect (k : Kernel) (post : BlockState → BlockState → Prop) : Prop :=
  Correct k.eraseDType post

/--
Algorithmic refinement for dtype-annotated kernels.

Both kernels are erased to the Real abstraction before checking the refinement
relation. This is the kernel-pair analogue of `AlgorithmCorrect`.
-/
def AlgorithmRefine
    (lhs rhs : Kernel)
    (rel : BlockState → BlockState → BlockState → Prop) : Prop :=
  Refine lhs.eraseDType rhs.eraseDType rel

/--
Computational correctness for observed floating outputs, expressed as an
epsilon bound against a mathematical specification.

This predicate is intentionally observation-level: `obs` may fail if the kernel
does not produce the expected output. It is the right shape for test-backed or
future IEEE-level claims; it is not used to prove Real algorithmic correctness.
-/
def ComputeCorrectAt?
    (k : Kernel) (ε : ℝ) (ι : Type)
    (spec : ι → BlockState → ℝ)
    (obs : ι → BlockState → Option ℝ) : Prop :=
  ∀ s s' i, exec k s = some s' →
    ∃ y, obs i s' = some y ∧ |y - spec i s| ≤ ε

/--
Computational refinement for observed floating outputs, expressed as an epsilon
bound between two kernels' observations.

This is the kernel-pair analogue of `ComputeCorrectAt?`: it is suitable for
test-backed or future IEEE-level claims about two runnable floating kernels.
-/
def ComputeRefineAt?
    (lhs rhs : Kernel) (ε : ℝ) (ι : Type)
    (lhsObs rhsObs : ι → BlockState → Option ℝ) : Prop :=
  ∀ s lhs' rhs' i,
    exec lhs s = some lhs' →
    exec rhs s = some rhs' →
    ∃ yL yR,
      lhsObs i lhs' = some yL ∧
      rhsObs i rhs' = some yR ∧
      |yL - yR| ≤ ε

/-- Transfer a real-kernel correctness theorem to an algorithmic theorem by
showing that floating-dtype erasure produces the real kernel. -/
theorem algorithmCorrect_of_erase_eq {k realK : Kernel}
    {post : BlockState → BlockState → Prop}
    (herase : k.eraseDType = realK)
    (hreal : Correct realK post) :
    AlgorithmCorrect k post := by
  intro s s' h
  exact hreal s s' (by simpa [AlgorithmCorrect, Correct, herase] using h)

/-- Transfer a real-kernel refinement theorem to an algorithmic refinement
theorem by showing that floating-dtype erasure produces the real kernels. -/
theorem algorithmRefine_of_erase_eq {lhs rhs realL realR : Kernel}
    {rel : BlockState → BlockState → BlockState → Prop}
    (hlhs : lhs.eraseDType = realL)
    (hrhs : rhs.eraseDType = realR)
    (hrefine : Refine realL realR rel) :
    AlgorithmRefine lhs rhs rel := by
  intro s lhs' rhs' hl hr
  exact hrefine s lhs' rhs'
    (by simpa [AlgorithmRefine, Refine, hlhs] using hl)
    (by simpa [AlgorithmRefine, Refine, hrhs] using hr)

end Kernel

namespace ComputeKernel

/-! ## Compute-kernel algorithm projection -/

/-- Public spec shape used by the current postcondition-style correctness API. -/
abbrev AlgSpec := BlockState → BlockState → Prop

/-- Algorithm correctness for compute kernels: successful projection exposes
ordinary `Kernel.Correct`; failed projection is intentionally unprovable. -/
def AlgorithmCorrect (ck : ComputeKernel) (post : AlgSpec) : Prop :=
  match ck.toAlgorithm? with
  | Except.ok ak => Kernel.Correct ak post
  | Except.error _ => False

/-- The projected algorithm proof obligation for a compute kernel. -/
def ProjectedCorrect (ck : ComputeKernel) (post : AlgSpec) : Prop :=
  AlgorithmCorrect ck post

/-- Public compute-facing correctness surface.

This is the Lean proof layer for `ComputeKernel`: the theorem starts from
compute-facing syntax, then projects through `toAlgorithm?` and proves the
algorithmic Real semantics. The optional `gap` records whether the
compute-to-algorithm gap is ignored or must be externally checked. -/
def ComputeCorrect
    (ck : ComputeKernel) (post : AlgSpec)
    (gap : GapPolicy := .ignore) : Prop :=
  GapPolicy.Holds gap ∧ ProjectedCorrect ck post

/-- The projected algorithm refinement obligation for a compute-kernel pair. -/
def ProjectedRefine
    (lhs rhs : ComputeKernel)
    (rel : BlockState → BlockState → BlockState → Prop) : Prop :=
  match lhs.toAlgorithm?, rhs.toAlgorithm? with
  | Except.ok lhsAlg, Except.ok rhsAlg => Kernel.Refine lhsAlg rhsAlg rel
  | _, _ => False

/-- Public compute-facing refinement surface. Both compute kernels must project
successfully to algorithm kernels, then the ordinary algorithm refinement proof
runs on the projected pair. The optional `gap` records whether the
compute-to-algorithm gap is ignored or must be externally checked. -/
def ComputeRefine
    (lhs rhs : ComputeKernel)
    (rel : BlockState → BlockState → BlockState → Prop)
    (gap : GapPolicy := .ignore) : Prop :=
  GapPolicy.Holds gap ∧ ProjectedRefine lhs rhs rel

/-- One-run compute-facing correctness from a fixed initial state.

This is the ergonomic surface for example theorems that already fix their input
state and loaded-input hypotheses. It packages the generic two-state
`ComputeCorrect` postcondition as `s0 = s -> post s'`. -/
def ExecCorrect
    (ck : ComputeKernel) (s : BlockState) (post : BlockState → Prop) : Prop :=
  ComputeCorrect ck (fun s0 s' => s0 = s → post s')

/-- One-run correctness surface for kernels whose observable output is a 1D
tensor view matching an array-shaped spec. -/
def OutputsMatchArray {n : Nat}
    (ck : ComputeKernel) (s : BlockState)
    (view : TensorView [n]) (spec : Fin n → ℝ) : Prop :=
  ExecCorrect ck s (fun s' => TensorView.matchesArray s' view spec)

/-- One-run compute-facing refinement from a fixed initial state. -/
def ExecRefine
    (lhs rhs : ComputeKernel) (s : BlockState)
    (rel : BlockState → BlockState → Prop) : Prop :=
  ComputeRefine lhs rhs (fun s0 lhs' rhs' => s0 = s → rel lhs' rhs')

/-- Current compute-kernel execution is defined by successful algorithm
projection. Future compute-only execution can refine this definition without
changing the `AlgorithmCorrect` API. -/
noncomputable def eval (ck : ComputeKernel) (s : BlockState) : Option BlockState :=
  match ck.toAlgorithm? with
  | Except.ok ak => exec ak s
  | Except.error _ => none

/-- Bridge from an explicit successful projection to compute-kernel algorithm
correctness. This keeps `Except` plumbing out of user theorem statements. -/
theorem algorithmCorrect_of_toAlgorithm_eq {ck : ComputeKernel} {ak : AlgKernel}
    {post : AlgSpec}
    (h : ck.toAlgorithm? = Except.ok ak)
    (hc : Kernel.Correct ak post) :
    AlgorithmCorrect ck post := by
  simp [AlgorithmCorrect, h, hc]

/-- Bridge from an explicit successful projection to public compute-facing
correctness. -/
theorem computeCorrect_of_toAlgorithm_eq {ck : ComputeKernel} {ak : AlgKernel}
    {post : AlgSpec}
    (h : ck.toAlgorithm? = Except.ok ak)
    (hc : Kernel.Correct ak post) :
    ComputeCorrect ck post := by
  exact ⟨trivial, algorithmCorrect_of_toAlgorithm_eq h hc⟩

/-- Build a compute-correctness theorem from an explicit gap obligation and
the projected algorithm proof. -/
theorem computeCorrect_of_gap_and_projected {ck : ComputeKernel} {post : AlgSpec}
    {gap : GapPolicy}
    (hgap : GapPolicy.Holds gap)
    (hproj : ProjectedCorrect ck post) :
    ComputeCorrect ck post gap := by
  exact ⟨hgap, hproj⟩

/-- Bridge from successful projection to compute correctness when the theorem
surface requires an externally checked gap contract. -/
theorem computeCorrect_of_toAlgorithm_eq_with_gap {ck : ComputeKernel}
    {ak : AlgKernel} {post : AlgSpec} {contract : ComputeGapContract}
    (hgap : ExternalChecked contract)
    (h : ck.toAlgorithm? = Except.ok ak)
    (hc : Kernel.Correct ak post) :
    ComputeCorrect ck post (gap := .require contract) := by
  exact ⟨hgap, algorithmCorrect_of_toAlgorithm_eq h hc⟩

/-- Bridge for projected algorithm kernels using `ck.toAlgKernel` as the
execution surface. This is the preferred bridge once examples are
compute-facing by default. -/
theorem computeCorrect_of_toAlgKernel {ck : ComputeKernel} {post : AlgSpec}
    (h : ck.toAlgorithm? = Except.ok ck.toAlgKernel)
    (hc : Kernel.Correct ck.toAlgKernel post) :
    ComputeCorrect ck post := by
  exact computeCorrect_of_toAlgorithm_eq h hc

/-- Projection failure makes algorithm correctness impossible. -/
theorem not_algorithmCorrect_of_toAlgorithm_error {ck : ComputeKernel}
    {err : EraseDTypeError} {post : AlgSpec}
    (h : ck.toAlgorithm? = Except.error err) :
    ¬ AlgorithmCorrect ck post := by
  simp [AlgorithmCorrect, h]

/-- Projection failure makes public compute correctness impossible. -/
theorem not_computeCorrect_of_toAlgorithm_error {ck : ComputeKernel}
    {err : EraseDTypeError} {post : AlgSpec}
    (h : ck.toAlgorithm? = Except.error err) :
    ¬ ComputeCorrect ck post := by
  intro hc
  exact not_algorithmCorrect_of_toAlgorithm_error h hc.2

/-- Bridge from explicit successful projections to public compute-facing
refinement. -/
theorem computeRefine_of_toAlgorithm_eq
    {lhs rhs : ComputeKernel} {lhsAlg rhsAlg : AlgKernel}
    {rel : BlockState → BlockState → BlockState → Prop}
    (hlhs : lhs.toAlgorithm? = Except.ok lhsAlg)
    (hrhs : rhs.toAlgorithm? = Except.ok rhsAlg)
    (hrefine : Kernel.Refine lhsAlg rhsAlg rel) :
    ComputeRefine lhs rhs rel := by
  exact ⟨trivial, by simp [ProjectedRefine, hlhs, hrhs, hrefine]⟩

/-- Build a compute-refinement theorem from an explicit gap obligation and the
projected algorithm refinement proof. -/
theorem computeRefine_of_gap_and_projected {lhs rhs : ComputeKernel}
    {rel : BlockState → BlockState → BlockState → Prop} {gap : GapPolicy}
    (hgap : GapPolicy.Holds gap)
    (hproj : ProjectedRefine lhs rhs rel) :
    ComputeRefine lhs rhs rel gap := by
  exact ⟨hgap, hproj⟩

/-- Bridge from successful projections to compute refinement when the theorem
surface requires an externally checked gap contract. -/
theorem computeRefine_of_toAlgorithm_eq_with_gap
    {lhs rhs : ComputeKernel} {lhsAlg rhsAlg : AlgKernel}
    {rel : BlockState → BlockState → BlockState → Prop}
    {contract : ComputeGapContract}
    (hgap : ExternalChecked contract)
    (hlhs : lhs.toAlgorithm? = Except.ok lhsAlg)
    (hrhs : rhs.toAlgorithm? = Except.ok rhsAlg)
    (hrefine : Kernel.Refine lhsAlg rhsAlg rel) :
    ComputeRefine lhs rhs rel (gap := .require contract) := by
  exact ⟨hgap, by simp [ProjectedRefine, hlhs, hrhs, hrefine]⟩

/-- Bridge for projected algorithm-kernel refinements using `toAlgKernel` on
both sides. -/
theorem computeRefine_of_toAlgKernel
    {lhs rhs : ComputeKernel}
    {rel : BlockState → BlockState → BlockState → Prop}
    (hlhs : lhs.toAlgorithm? = Except.ok lhs.toAlgKernel)
    (hrhs : rhs.toAlgorithm? = Except.ok rhs.toAlgKernel)
    (hrefine : Kernel.Refine lhs.toAlgKernel rhs.toAlgKernel rel) :
    ComputeRefine lhs rhs rel := by
  exact computeRefine_of_toAlgorithm_eq hlhs hrhs hrefine

/-- A projection failure on either side makes compute refinement impossible. -/
theorem not_computeRefine_of_left_toAlgorithm_error {lhs rhs : ComputeKernel}
    {err : EraseDTypeError}
    {rel : BlockState → BlockState → BlockState → Prop}
    (h : lhs.toAlgorithm? = Except.error err) :
    ¬ ComputeRefine lhs rhs rel := by
  intro hrefine
  have hproj := hrefine.2
  cases hrhs : rhs.toAlgorithm? <;> simp [ProjectedRefine, h, hrhs] at hproj

/-- A projection failure on either side makes compute refinement impossible. -/
theorem not_computeRefine_of_right_toAlgorithm_error {lhs rhs : ComputeKernel}
    {err : EraseDTypeError}
    {rel : BlockState → BlockState → BlockState → Prop}
    (h : rhs.toAlgorithm? = Except.error err) :
    ¬ ComputeRefine lhs rhs rel := by
  intro hrefine
  have hproj := hrefine.2
  cases hlhs : lhs.toAlgorithm? <;> simp [ProjectedRefine, h, hlhs] at hproj

/-- One-run correctness bridge for compute-facing kernels through their
projected algorithm kernel. -/
theorem execCorrect_of_toAlgKernel {ck : ComputeKernel} {s : BlockState}
    {post : BlockState → Prop}
    (h : ck.toAlgorithm? = Except.ok ck.toAlgKernel)
    (hc : ∀ s', exec ck.toAlgKernel s = some s' → post s') :
    ExecCorrect ck s post := by
  apply computeCorrect_of_toAlgKernel h
  intro s0 s' hExec hs0
  subst s0
  exact hc s' hExec

/-- Projection failure makes one-run compute correctness impossible. -/
theorem not_execCorrect_of_toAlgorithm_error {ck : ComputeKernel}
    {err : EraseDTypeError} {s : BlockState} {post : BlockState → Prop}
    (h : ck.toAlgorithm? = Except.error err) :
    ¬ ExecCorrect ck s post := by
  intro hc
  exact not_computeCorrect_of_toAlgorithm_error h hc

/-- One-run refinement bridge for compute-facing kernels through their
projected algorithm kernels. -/
theorem execRefine_of_toAlgKernel {lhs rhs : ComputeKernel} {s : BlockState}
    {rel : BlockState → BlockState → Prop}
    (hlhs : lhs.toAlgorithm? = Except.ok lhs.toAlgKernel)
    (hrhs : rhs.toAlgorithm? = Except.ok rhs.toAlgKernel)
    (hrefine : ∀ lhs' rhs',
      exec lhs.toAlgKernel s = some lhs' →
      exec rhs.toAlgKernel s = some rhs' →
      rel lhs' rhs') :
    ExecRefine lhs rhs s rel := by
  apply computeRefine_of_toAlgKernel hlhs hrhs
  intro s0 lhs' rhs' hL hR hs0
  subst s0
  exact hrefine lhs' rhs' hL hR

/-- Definition-unfolding lemma: when projection succeeds, `ComputeKernel.eval`
unfolds to `exec` on the projected algorithm kernel.

This is **not** a compute-vs-algorithm soundness theorem. `ComputeKernel.eval`
is defined as `match ck.toAlgorithm? with | Except.ok ak => exec ak s | _ => none`,
so the equation here is just the definition unfolded for the `ok` case. It is
provided as a `simp`-friendly convenience.

Compute-layer semantics (IEEE rounding, NaN, denormals, etc.) are verified
empirically by the differential testing pipeline (see PLAN.md "Verification
architecture" and `documents/EraseDType.md`). There is, by design, no Lean
theorem connecting compute execution to algorithm execution. -/
theorem eval_eq_exec_of_toAlgorithm? {ck : ComputeKernel} {ak : AlgKernel}
    (h : ck.toAlgorithm? = Except.ok ak) :
    ∀ s, ck.eval s = exec ak s := by
  intro s
  simp [eval, h]

end ComputeKernel

end VeriTile.Triton
