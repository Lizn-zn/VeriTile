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

namespace ComputeCorrect

/-! ## User-facing realization combinators -/

/-- Fully general two-state correctness surface. Prefer `Post` or a `Realizes`
combinator when the theorem fixes one initial state. -/
def General (ck : ComputeKernel) (post : ComputeKernel.AlgSpec) : Prop :=
  ComputeKernel.ComputeCorrect ck post

/-- One-run postcondition-style correctness from a fixed initial state. -/
def Post
    (ck : ComputeKernel) (s : BlockState) (post : BlockState → Prop) : Prop :=
  ComputeKernel.ExecCorrect ck s post

/--
Logical output write map indexed by `ι`.

The map is intentionally state-free: it describes which memory cells are
written by each logical output index, while the execution state is supplied
separately to `Realizes`. A `some addr` lane is checked against `expected`; a
`none` lane is inactive and imposes no output obligation.
-/
abbrev WriteMap (ι : Type) := ι → Option MemCellAddr

namespace WriteMap

/-- Build an output target from an existing tensor view. -/
def ofTensorView {shape : TileShape} (view : TensorView shape) :
    WriteMap (TileIndex shape) :=
  fun idx => some (view.region, view.offset idx)

/-- Build a scalar output target for one memory cell. -/
def scalar (region : RegionName) (offset : Nat) : WriteMap PUnit :=
  fun _ => some (region, offset)

/-- Build a write map from a mask and an address map. -/
def writeIf {ι : Type} (mask : ι → Prop) [DecidablePred mask]
    (addr : ι → MemCellAddr) : WriteMap ι :=
  fun i => if mask i then some (addr i) else none

end WriteMap

/--
A contiguous 1D output tile in one memory region.

Lane `i : Fin size` writes to `base + i`; the lane is active exactly when
`base + i < bound`. This captures the common Triton pattern
`offsets = base + tl.arange(0, BLOCK); mask = offsets < N`.
-/
structure TileBlock where
  region : RegionName
  base   : Nat
  size   : Nat
  bound  : Nat

namespace TileBlock

/-- Standard `ComputeCorrect.Realizes` write map for this block. -/
def writes (t : TileBlock) : WriteMap (Fin t.size) :=
  WriteMap.writeIf (fun i : Fin t.size => t.base + i.val < t.bound)
    (fun i => (t.region, t.base + i.val))

@[simp] theorem writes_apply_active (t : TileBlock) (i : Fin t.size)
    (h : t.base + i.val < t.bound) :
    t.writes i = some (t.region, t.base + i.val) := by
  simp [writes, WriteMap.writeIf, h]

@[simp] theorem writes_apply_inactive (t : TileBlock) (i : Fin t.size)
    (h : ¬ t.base + i.val < t.bound) :
    t.writes i = none := by
  simp [writes, WriteMap.writeIf, h]

end TileBlock

/--
Readable output carrier for overloaded output surfaces.

The `MemCell` instance is exact algorithm-layer cell equality. The `ℝ` and
`Nat` instances use the corresponding readback projections.
-/
class OutputReadable (α : Type) where
  read : BlockState → MemCellAddr → α

instance : OutputReadable MemCell where
  read s addr := s.mem addr.1 addr.2

instance : OutputReadable ℝ where
  read s addr := s.readMem addr.1 addr.2

instance : OutputReadable Nat where
  read s addr := s.readMemValue .nat addr.1 addr.2

instance : OutputReadable Int where
  read s addr := s.readMemValue .int addr.1 addr.2

@[simp] theorem OutputReadable.read_memcell (s : BlockState) (addr : MemCellAddr) :
    (OutputReadable.read s addr : MemCell) = s.mem addr.1 addr.2 := rfl

@[simp] theorem OutputReadable.read_real (s : BlockState) (addr : MemCellAddr) :
    (OutputReadable.read s addr : ℝ) = s.readMem addr.1 addr.2 := rfl

@[simp] theorem OutputReadable.read_nat (s : BlockState) (addr : MemCellAddr) :
    (OutputReadable.read s addr : Nat) = s.readMemValue .nat addr.1 addr.2 := rfl

@[simp] theorem OutputReadable.read_int (s : BlockState) (addr : MemCellAddr) :
    (OutputReadable.read s addr : Int) = s.readMemValue .int addr.1 addr.2 := rfl

/--
Overloaded correctness-realization surface over a state-free output write map.

The carrier is inferred from `expected`: `ℝ` uses `readMem`, `Nat` uses
`readMemValue .nat`, and `MemCell` uses exact algorithm-layer cell equality.
-/
def Realizes {ι : Type} {α : Type} [OutputReadable α]
    (kernel : ComputeKernel) (initialState : BlockState)
    (write : WriteMap ι) (expected : ι → α) : Prop :=
  ComputeKernel.ExecCorrect kernel initialState (fun s' =>
    ∀ i : ι, match write i with
      | some addr => OutputReadable.read s' addr = expected i
      | none => True)

@[simp] theorem realizes_writeIf_iff {ι : Type} {α : Type} [OutputReadable α]
    (kernel : ComputeKernel) (initialState : BlockState)
    (mask : ι → Prop) [DecidablePred mask]
    (addr : ι → MemCellAddr) (expected : ι → α) :
    Realizes kernel initialState (WriteMap.writeIf mask addr) expected ↔
      ComputeKernel.ExecCorrect kernel initialState (fun s' =>
        ∀ i : ι, mask i → OutputReadable.read s' (addr i) = expected i) := by
  unfold Realizes WriteMap.writeIf ComputeKernel.ExecCorrect
    ComputeKernel.ComputeCorrect ComputeKernel.ProjectedCorrect
    ComputeKernel.AlgorithmCorrect Kernel.Correct
  cases hAlg : kernel.toAlgorithm? with
  | error e =>
      simp
  | ok ak =>
      simp
      constructor
      · intro h s0 s' hExec hs0 i hi
        have hout := h s0 s' hExec hs0 i
        simpa [hi] using hout
      · intro h s0 s' hExec hs0 i
        by_cases hi : mask i
        · simpa [hi] using h s0 s' hExec hs0 i hi
        · simp [hi]

/-- A scalar Real output equals `expected` after executing `kernel` from `initialState`. -/
def OutputScalar
    (kernel : ComputeKernel) (initialState : BlockState)
    (out : RegionName) (offset : Nat) (expected : ℝ) : Prop :=
  Realizes kernel initialState
    (fun _ : PUnit => some (out, offset))
    (fun _ => expected)

/-- A scalar Nat output equals `expected` after executing `kernel` from `initialState`. -/
def OutputNatScalar
    (kernel : ComputeKernel) (initialState : BlockState)
    (out : RegionName) (offset : Nat) (expected : Nat) : Prop :=
  Realizes kernel initialState
    (fun _ : PUnit => some (out, offset))
    (fun _ => expected)

@[simp] theorem OutputScalar_iff
    (ck : ComputeKernel) (s : BlockState) (out : RegionName) (offset : Nat) (e : ℝ) :
    OutputScalar ck s out offset e ↔
      ComputeKernel.ExecCorrect ck s (fun s' => s'.readMem out offset = e) := by
  simp [OutputScalar, Realizes]

@[simp] theorem OutputNatScalar_iff
    (ck : ComputeKernel) (s : BlockState) (out : RegionName) (offset : Nat) (e : Nat) :
    OutputNatScalar ck s out offset e ↔
      ComputeKernel.ExecCorrect ck s (fun s' => s'.readMemValue .nat out offset = e) := by
  simp [OutputNatScalar, Realizes]

/-- An ND Real tensor view matches `expected` after executing `ck` from `s`. -/
def OutputTile {shape : TileShape}
    (ck : ComputeKernel) (s : BlockState)
    (view : TensorView shape) (expected : TileIndex shape → ℝ) : Prop :=
  Realizes ck s (WriteMap.ofTensorView view) expected

/-- A 1D Real tensor view matches `expected` after executing `ck` from `s`. -/
def OutputArray {n : Nat}
    (ck : ComputeKernel) (s : BlockState)
    (view : TensorView [n]) (expected : Fin n → ℝ) : Prop :=
  OutputTile ck s view (fun idx : TileIndex [n] => expected idx.1)

/--
A paired 1D value/index output matches expected values on active lanes.

This is the user-facing shape for kernels such as `tl.max(...,
return_indices=True)`: the value channel is Real memory, while the index channel
is typed Nat memory.

Note: this is a documented heterogeneous-carrier special case. The `Realizes`
surface dispatches on a single `α` per call; pair outputs need two readback
types simultaneously (`ℝ` for value, `Nat` for index), so they keep a dedicated
definition rather than routing through `Realizes`. A future revision could
introduce a per-lane carrier typeclass to subsume this; not worth doing until
a second heterogeneous-output kernel appears.
-/
def OutputPairWhere {n : Nat}
    (ck : ComputeKernel) (s : BlockState)
    (valueRegion indexRegion : RegionName)
    (offset : Fin n → Nat)
    (active : Fin n → Prop)
    (expectedValue : Fin n → ℝ)
    (expectedIndex : Fin n → Nat) : Prop :=
  ComputeKernel.ExecCorrect ck s (fun s' =>
    ∀ i : Fin n,
      active i →
      s'.readMem valueRegion (offset i) = expectedValue i ∧
      s'.readMemValue .nat indexRegion (offset i) = expectedIndex i)

/-- A paired 1D value/index output matches expected values on every lane. -/
def OutputPair {n : Nat}
    (ck : ComputeKernel) (s : BlockState)
    (valueRegion indexRegion : RegionName)
    (offset : Fin n → Nat)
    (expectedValue : Fin n → ℝ)
    (expectedIndex : Fin n → Nat) : Prop :=
  OutputPairWhere ck s valueRegion indexRegion offset (fun _ => True)
    expectedValue expectedIndex

end ComputeCorrect

namespace ComputeRefine

/-! ## User-facing refinement realization combinators -/

/-- Fully general two-state refinement surface. Prefer `Post` or a `Realizes`
combinator when the theorem fixes one initial state. -/
def General
    (lhs rhs : ComputeKernel)
    (rel : BlockState → BlockState → BlockState → Prop) : Prop :=
  ComputeKernel.ComputeRefine lhs rhs rel

/-- One-run postcondition-style refinement from a fixed initial state. -/
def Post
    (lhs rhs : ComputeKernel) (s : BlockState)
    (rel : BlockState → BlockState → Prop) : Prop :=
  ComputeKernel.ExecRefine lhs rhs s rel

/--
Overloaded refinement-realization surface over state-free output write maps.

The left and right kernels may write to different target cells. The carrier
types are inferred independently from `relation`; use the same write map on
both sides for ordinary same-buffer equivalence. Lanes where either side is
`none` impose no output relation.
-/
def Realizes {ι : Type} {α β : Type}
    [ComputeCorrect.OutputReadable α] [ComputeCorrect.OutputReadable β]
    (lhs rhs : ComputeKernel) (initialState : BlockState)
    (lhsWrite rhsWrite : ComputeCorrect.WriteMap ι)
    (relation : ι → α → β → Prop) : Prop :=
  ComputeKernel.ExecRefine lhs rhs initialState (fun lhs' rhs' =>
    ∀ i : ι, match lhsWrite i, rhsWrite i with
      | some lhsAddr, some rhsAddr =>
          relation i
            (ComputeCorrect.OutputReadable.read lhs' lhsAddr)
            (ComputeCorrect.OutputReadable.read rhs' rhsAddr)
      | _, _ => True)

/-- Two kernels produce equal Real scalar observations after running from `s`. -/
def OutputScalarEq
    (lhs rhs : ComputeKernel) (s : BlockState)
    (lhsOut : RegionName) (lhsOffset : Nat)
    (rhsOut : RegionName) (rhsOffset : Nat) : Prop :=
  ComputeKernel.ExecRefine lhs rhs s (fun lhs' rhs' =>
    lhs'.readMem lhsOut lhsOffset = rhs'.readMem rhsOut rhsOffset)

/-- Two kernels produce equal Nat scalar observations after running from `s`. -/
def OutputNatScalarEq
    (lhs rhs : ComputeKernel) (s : BlockState)
    (lhsOut : RegionName) (lhsOffset : Nat)
    (rhsOut : RegionName) (rhsOffset : Nat) : Prop :=
  ComputeKernel.ExecRefine lhs rhs s (fun lhs' rhs' =>
    lhs'.readMemValue .nat lhsOut lhsOffset =
      rhs'.readMemValue .nat rhsOut rhsOffset)

/-- Two kernels produce equal ND Real tensor-view observations. -/
def OutputTileEq {shape : TileShape}
    (lhs rhs : ComputeKernel) (s : BlockState)
    (lhsView rhsView : TensorView shape) : Prop :=
  ComputeKernel.ExecRefine lhs rhs s (fun lhs' rhs' =>
    ∀ idx : TileIndex shape,
      TensorView.observe (some lhs') lhsView idx =
        TensorView.observe (some rhs') rhsView idx)

/-- Two kernels produce equal 1D Real tensor-view observations. -/
def OutputArrayEq {n : Nat}
    (lhs rhs : ComputeKernel) (s : BlockState)
    (lhsView rhsView : TensorView [n]) : Prop :=
  OutputTileEq lhs rhs s lhsView rhsView

/-- Two kernels produce equal paired value/index outputs on active lanes. -/
def OutputPairEqWhere {n : Nat}
    (lhs rhs : ComputeKernel) (s : BlockState)
    (lhsValue lhsIndex rhsValue rhsIndex : RegionName)
    (lhsOffset rhsOffset : Fin n → Nat)
    (active : Fin n → Prop) : Prop :=
  ComputeKernel.ExecRefine lhs rhs s (fun lhs' rhs' =>
    ∀ i : Fin n,
      active i →
      lhs'.readMem lhsValue (lhsOffset i) =
          rhs'.readMem rhsValue (rhsOffset i) ∧
        lhs'.readMemValue .nat lhsIndex (lhsOffset i) =
          rhs'.readMemValue .nat rhsIndex (rhsOffset i))

/-- Two kernels produce equal paired value/index outputs on every lane. -/
def OutputPairEq {n : Nat}
    (lhs rhs : ComputeKernel) (s : BlockState)
    (lhsValue lhsIndex rhsValue rhsIndex : RegionName)
    (lhsOffset rhsOffset : Fin n → Nat) : Prop :=
  OutputPairEqWhere lhs rhs s lhsValue lhsIndex rhsValue rhsIndex
    lhsOffset rhsOffset (fun _ => True)

end ComputeRefine

end VeriTile.Triton
