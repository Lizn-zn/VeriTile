import VeriTile.Triton
import VeriTile.Examples.Common
import VeriTile.Meta.StatementAudit

/-!
# Float dtype erasure — the float-facing *refinement* policy

Self-contained showcase, read top to bottom: the **kernels** first (what we
built), the **supporting lemmas** in the middle (`private` plumbing — the
erasure bridges and the exec-level view), the **theorem** last (one public
headline `float_softmax_reciprocal_refinement_view`), then a compile-time
**trust audit**. The three real sections below are
`FloatDTypeRefine.kernels`, `FloatDTypeRefine.lemmas`,
`FloatDTypeRefine.theorems`.

VeriTile's float-facing theorem policy on the **refinement** surface
(`ComputeRefine`): two kernels carry explicit `tl.float32` memory annotations,
and the algorithmic equivalence theorem is discharged by **erasing** those
annotations to the Real channel and reusing the Real-valued refinement (here
the softmax reciprocal rewrite). This is the erased/ideal pathway — the
complement of the rounding-model pathway (`ComputeRefine.*R`) demonstrated in
`bench/examples/FusedSwigluEquiv.lean`. The correctness counterpart of this policy
lives in `bench/examples/FloatDTypeCorrect.lean`.

## The public result (bottom of file)

The single public headline is **`float_softmax_reciprocal_refinement_view`** —
the erased equivalence statement (`ComputeRefine.General`): the fp32-annotated
per-element-divide and reciprocal-form softmaxes agree cell-by-cell after
erasure. Its statement's project surface is pinned by the `#stmtSurfaceSubset`
gate below — **no spec** (this is a refinement surface). The erasure bridges
and the exec-level view are `private` scaffolding.
-/

namespace VeriTile.Bench.Examples.FloatDTypeRefine

open VeriTile.Triton VeriTile.Triton.TiledSoftmax
open VeriTile.Examples (programTileView InputLoadedAt
  inputLoadedAt_of_programTileView_loaded observeAt injective_offset_singleton)

/-- Local reduction helper: the algorithm-projection traversals
(`ComputeStmt.toAlgorithm?` and friends) are written in `do` notation over
`Except`, so their equation lemmas leave `Except.ok _ >>= f` redexes behind.
This discharges them for `simp`. -/
@[simp] private theorem except_ok_bind {α β ε : Type _} (a : α)
    (f : α → Except ε β) :
    (Except.ok a : Except ε α) >>= f = f a := rfl

/-! ## Kernels -/
section FloatDTypeRefine.kernels

/-- This file's own copy of the aligned Real per-element-divide stable softmax
(each showcase is self-contained; the bf16 sibling is showcased in
`bench/examples/SoftmaxStableEquiv.lean` — its raw-store twin was retired
there, the exact surface being the `≡[R]` headline at `R := .triv`). The
fp32-annotated `floatStableSoftmaxKernel` below erases to this kernel. -/
def stableSoftmaxKernel (xReg yReg : RegionName) (blockSize : Nat) : ComputeKernel := triton {
  pid  := tl.program_id(0)
  offs := pid * $(blockSize) + tl.arange(0, $(blockSize))
  x    := tl.load($(xReg) + offs)
  m    := tl.max(x, axis=0)
  e    := tl.exp(x - m)
  s    := tl.sum(e, axis=0)
  y    := e / s
  tl.store($(yReg) + offs, y)
}

/-- This file's own copy of the aligned Real reciprocal-form stable softmax
(the bf16 sibling is showcased in `bench/examples/SoftmaxReciprocalEquiv.lean`;
its raw-store twin was retired there). The fp32-annotated
`floatSoftmaxRecipKernel` below erases to this kernel. -/
def softmaxRecipKernel (xReg yReg : RegionName) (blockSize : Nat) : ComputeKernel := triton {
  pid    := tl.program_id(0)
  offs   := pid * $(blockSize) + tl.arange(0, $(blockSize))
  x      := tl.load($(xReg) + offs)
  m      := tl.max(x, axis=0)
  e      := tl.exp(x - m)
  s      := tl.sum(e, axis=0)
  inv_s  := 1 / s
  y      := e * inv_s
  tl.store($(yReg) + offs, y)
}

/-- Stable softmax with explicit fp32 input/output annotations and per-element
division `y = e / s`. The reductions still run in the Real abstraction after
the input load is cast to `tl.float64`. -/
def floatStableSoftmaxKernel (xReg yReg : RegionName) (blockSize : Nat) : ComputeKernel := triton {
  pid  := tl.program_id(0)
  offs := pid * $(blockSize) + tl.arange(0, $(blockSize))
  x    := (tl.load($(xReg) + offs, dtype=tl.float32)).to(tl.float64)
  m    := tl.max(x, axis=0)
  e    := tl.exp(x - m)
  s    := tl.sum(e, axis=0)
  y    := e / s
  tl.store($(yReg) + offs, (y).to(tl.float32))
}

/-- Optimized stable softmax: precompute `1 / s` and use multiplication,
saving per-lane divisions versus `floatStableSoftmaxKernel`. -/
def floatSoftmaxRecipKernel (xReg yReg : RegionName) (blockSize : Nat) : ComputeKernel := triton {
  pid    := tl.program_id(0)
  offs   := pid * $(blockSize) + tl.arange(0, $(blockSize))
  x      := (tl.load($(xReg) + offs, dtype=tl.float32)).to(tl.float64)
  m      := tl.max(x, axis=0)
  e      := tl.exp(x - m)
  s      := tl.sum(e, axis=0)
  inv_s  := 1 / s
  y      := e * inv_s
  tl.store($(yReg) + offs, (y).to(tl.float32))
}

end FloatDTypeRefine.kernels

/-! ## Supporting lemmas (private plumbing) -/
section FloatDTypeRefine.lemmas

/-- The fp32 per-element-divide softmax erases to the existing Real kernel,
at the algorithm-projection level. -/
private theorem float_stable_softmax_erases_to_real
    (xReg yReg : RegionName) (blockSize : Nat) :
    (floatStableSoftmaxKernel xReg yReg blockSize).eraseDType.toAlgKernel =
      (stableSoftmaxKernel xReg yReg blockSize).toAlgKernel := by
  simp [floatStableSoftmaxKernel, stableSoftmaxKernel, ComputeKernel.eraseDType,
    ComputeStmt.toAlgorithm?, ComputeStmt.listToAlgorithm?,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, ComputeDType.eraseDType,
    Kernel.eraseDType, Stmt.eraseDTypeList, Stmt.eraseDType,
    Op.eraseDType, VeriTile.Triton.eraseDType, NumericDType.eraseDType]
  repeat' apply And.intro
  all_goals
    rw [Op.eraseDType.eq_def]
    simp [MemAccess.eraseDType.eq_def, MaskOpt.eraseDType.eq_def,
      Op.eraseDType.eq_def, VeriTile.Triton.eraseDType]

/-- The fp32 reciprocal-form softmax erases to the existing Real optimized
kernel, at the algorithm-projection level. -/
private theorem float_softmax_recip_erases_to_real
    (xReg yReg : RegionName) (blockSize : Nat) :
    (floatSoftmaxRecipKernel xReg yReg blockSize).eraseDType.toAlgKernel =
      (softmaxRecipKernel xReg yReg blockSize).toAlgKernel := by
  simp [floatSoftmaxRecipKernel, softmaxRecipKernel, ComputeKernel.eraseDType,
    ComputeStmt.toAlgorithm?, ComputeStmt.listToAlgorithm?,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, ComputeDType.eraseDType,
    Kernel.eraseDType, Stmt.eraseDTypeList, Stmt.eraseDType,
    Op.eraseDType, VeriTile.Triton.eraseDType, NumericDType.eraseDType]
  repeat' apply And.intro
  all_goals
    rw [Op.eraseDType.eq_def]
    simp [MemAccess.eraseDType.eq_def, MaskOpt.eraseDType.eq_def,
      Op.eraseDType.eq_def, VeriTile.Triton.eraseDType]

/-! ### Local copy of the Real reciprocal refinement

The erased equivalence reduces to the Real-valued reciprocal rewrite
(`e / S = e · S⁻¹`). The bf16 sibling of this chain is showcased in
`bench/examples/SoftmaxReciprocalEquiv.lean` (whose exact-ℝ companion was
retired — the exact surface there is the `≡[R]` headline at `R := .triv`);
each showcase is self-contained, so this file carries its own `private` copy.
The math denotation (`stableSpec`, `tileMax`) lives in
`VeriTile.Triton.Math.Softmax`. -/

/-- The load-bearing math identity: division equals multiplication by
    reciprocal (for non-zero divisor). -/
private theorem div_eq_mul_inv_real (a s : ℝ) (hs : s ≠ 0) : a / s = a * (1 / s) := by
  field_simp

/-- Closed-form spec for `softmaxRecipKernel`'s `Y[pid*N+i]` cell. -/
private noncomputable def stableRecipSpec {N : Nat} (xs : Fin N → ℝ) (m : ℝ) (i : Fin N) : ℝ :=
  Real.exp (xs i - m) * (1 / ∑ j, Real.exp (xs j - m))

/-- Closed form of the divide-side kernel (this file's private copy; the
formerly-shared original was retired with SoftmaxStableEquiv's exact layer). -/
private theorem softmax_stable_correct
    (xReg yReg : RegionName)
    (blockSize : Nat) (hN : 0 < blockSize) (s : BlockState) (xs : Fin blockSize → ℝ)
    (_h_x : InputLoadedAt s xReg blockSize xs) :
    ∀ i : Fin blockSize,
      observeAt (exec (stableSoftmaxKernel xReg yReg blockSize) s) yReg blockSize s.pid i
        = some (stableSpec xs (tileMax hN xs) i) := by
  obtain ⟨n, rfl⟩ := Nat.exists_eq_succ_of_ne_zero hN.ne'
  intro i
  have h_inj : Function.Injective
      (fun idx : TileIndex [n + 1] => s.pid * (n + 1) + idx.1.val) := by
    rintro ⟨a, _⟩ ⟨b, _⟩ hab
    obtain rfl : a = b := Fin.ext (Nat.add_left_cancel hab)
    rfl
  simp [observeAt, exec, stableSoftmaxKernel, stepStmts, stepStmt, evalOp,
        Tile.bop, Tile.uop, Tile.reduceSum, Tile.reduceSumDrop,
        Tile.reduceMax, Tile.reduceMaxDrop,
        TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
        NumericDType.add, NumericDType.mul, NumericDType.sub, NumericDType.div, stableSpec, tileMax]
  repeat unfold evalOp
  simp [observeAt, exec, stableSoftmaxKernel, stepStmts, stepStmt,
        Tile.bop, Tile.uop, Tile.reduceSum, Tile.reduceSumDrop,
        Tile.reduceMax, Tile.reduceMaxDrop,
        TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
        NumericDType.add, NumericDType.mul, NumericDType.sub, NumericDType.div, stableSpec, tileMax]
  rw [BlockState.scatter_readback_nd _ _ _ h_inj (i, PUnit.unit)]
  unfold InputLoadedAt at _h_x
  simp [_h_x]
  rfl

/-- Closed form of the reciprocal-side kernel (this file's private copy; the
formerly-shared original was retired with SoftmaxReciprocalEquiv's exact
layer). -/
private theorem softmax_recip_correct
    (xReg yReg : RegionName)
    (N : Nat) (hN : 0 < N) (s : BlockState) (xs : Fin N → ℝ)
    (_h_x : InputLoadedAt s xReg N xs) :
    ∀ i : Fin N,
      observeAt (exec (softmaxRecipKernel xReg yReg N) s) yReg N s.pid i
        = some (stableRecipSpec xs (tileMax hN xs) i) := by
  obtain ⟨n, rfl⟩ := Nat.exists_eq_succ_of_ne_zero hN.ne'
  intro i
  have h_inj : Function.Injective
      (fun idx : TileIndex [n + 1] => s.pid * (n + 1) + idx.1.val) :=
    injective_offset_singleton (s.pid * (n + 1))
  simp [observeAt, exec, softmaxRecipKernel, stepStmts, stepStmt, evalOp,
        Tile.bop, Tile.uop, Tile.reduceSum, Tile.reduceSumDrop,
        Tile.reduceMax, Tile.reduceMaxDrop,
        TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
        NumericDType.add, NumericDType.mul, NumericDType.sub, NumericDType.div, stableRecipSpec, tileMax]
  repeat unfold evalOp
  simp [observeAt, exec, softmaxRecipKernel, stepStmts, stepStmt,
        Tile.bop, Tile.uop, Tile.reduceSum, Tile.reduceSumDrop,
        Tile.reduceMax, Tile.reduceMaxDrop,
        TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
        NumericDType.add, NumericDType.mul, NumericDType.sub, NumericDType.div, stableRecipSpec, tileMax]
  unfold InputLoadedAt at _h_x
  rw [BlockState.scatter_readback_nd _ _ _ h_inj (i, PUnit.unit)]
  simp [_h_x]
  rfl

/-- Refinement (exact ℝ): per-element division ≡ precomputed reciprocal.
Composes the two closed forms via `div_eq_mul_inv_real`. -/
private theorem softmax_reciprocal_refinement
    (xReg yReg : RegionName)
    (N : Nat) (hN : 0 < N) (s : BlockState) (xs : Fin N → ℝ)
    (h_x : InputLoadedAt s xReg N xs) :
    ∀ i : Fin N,
      observeAt (exec (stableSoftmaxKernel xReg yReg N) s) yReg N s.pid i =
      observeAt (exec (softmaxRecipKernel  xReg yReg N) s) yReg N s.pid i := by
  intro i
  rw [softmax_stable_correct xReg yReg N hN s xs h_x i,
      softmax_recip_correct  xReg yReg N hN s xs h_x i]
  congr 1
  -- Goal: stableSpec xs (tileMax hN xs) i = stableRecipSpec xs (tileMax hN xs) i
  -- These differ only by a / b vs a * (1/b).
  unfold stableSpec stableRecipSpec
  have h_sum_pos : 0 < ∑ j, Real.exp (xs j - tileMax hN xs) := by
    apply Finset.sum_pos
    · intro j _; exact Real.exp_pos _
    · obtain ⟨n, rfl⟩ := Nat.exists_eq_succ_of_ne_zero hN.ne'
      exact ⟨⟨0, hN⟩, Finset.mem_univ _⟩
  exact div_eq_mul_inv_real _ _ (ne_of_gt h_sum_pos)

/-- View-level surface for `softmax_reciprocal_refinement`. -/
private theorem softmax_reciprocal_refinement_exec_view
    (xReg yReg : RegionName)
    (N : Nat) (hN : 0 < N) (s : BlockState) (xs : Fin N → ℝ)
    (h_x : TensorView.loaded s (programTileView s xReg N)
      (fun idx : TileIndex [N] => xs idx.1)) :
    ∀ idx : TileIndex [N],
      TensorView.observe (exec (stableSoftmaxKernel xReg yReg N) s)
          (programTileView s yReg N) idx =
      TensorView.observe (exec (softmaxRecipKernel  xReg yReg N) s)
          (programTileView s yReg N) idx := by
  intro idx
  have hx := inputLoadedAt_of_programTileView_loaded (s := s) (region := xReg)
    (N := N) (xs := xs) h_x
  simpa [TensorView.observe, observeTileAt, programTileView,
         TensorView.offset, Offset.strided, observeAt]
    using softmax_reciprocal_refinement xReg yReg N hN s xs hx idx.1

/-- Float-facing rewrite refinement, exec-level view: the fp32
per-element-divide softmax and the fp32 reciprocal-form softmax are
algorithmically equivalent after erasure. State the dtype-annotated rewrite,
prove it through the existing Real refinement. -/
private theorem float_softmax_reciprocal_refinement_exec_view
    (xReg yReg : RegionName)
    (N : Nat) (hN : 0 < N) (s : BlockState) (xs : Fin N → ℝ)
    (h_x : TensorView.loaded s (programTileView s xReg N)
      (fun idx : TileIndex [N] => xs idx.1)) :
    ∀ idx : TileIndex [N],
      TensorView.observe
          (exec (floatStableSoftmaxKernel xReg yReg N).eraseDType s)
          (programTileView s yReg N) idx =
      TensorView.observe
          (exec (floatSoftmaxRecipKernel xReg yReg N).eraseDType s)
          (programTileView s yReg N) idx := by
  intro idx
  rw [float_stable_softmax_erases_to_real xReg yReg N,
    float_softmax_recip_erases_to_real xReg yReg N]
  exact softmax_reciprocal_refinement_exec_view xReg yReg N hN s xs h_x idx

end FloatDTypeRefine.lemmas

/-! ## The headline theorem -/
section FloatDTypeRefine.theorems

/- Shared parameters of the headline, hoisted to a `variable` block so the
signature carries only its input hypothesis. (The input keeps the `loaded`
view form here because the reused Real reciprocal refinement expects it.) -/
variable (xReg yReg : RegionName) (N : Nat) (hN : 0 < N) (s : BlockState) (xs : Fin N → ℝ)

include hN in
/-- **float divide refines float reciprocal** (`ComputeRefine.General`, erased
view): the two fp32-annotated softmax kernels, projected to their erased
algorithm kernels, perform the same per-lane writes. Proved through the Real
reciprocal refinement via the erasure bridges. -/
specification float_softmax_reciprocal_refinement_view
    (h_x : TensorView.loaded s (programTileView s xReg N)
      (fun idx : TileIndex [N] => xs idx.1)) :
    ComputeRefine.General
      (((floatStableSoftmaxKernel xReg yReg N).eraseDType))
      (((floatSoftmaxRecipKernel xReg yReg N).eraseDType))
      (fun s0 lhs' rhs' =>
        s0 = s →
        ∀ idx : TileIndex [N],
          TensorView.observe (some lhs')
              (programTileView s yReg N) idx =
          TensorView.observe (some rhs')
              (programTileView s yReg N) idx) := by
  apply ComputeKernel.computeRefine_of_toAlgKernel
    (by simp [ComputeKernel.eraseDType]) (by simp [ComputeKernel.eraseDType])
  intro s0 lhs' rhs' hL hR hs0
  subst s0
  intro idx
  have hview := float_softmax_reciprocal_refinement_exec_view xReg yReg N hN s xs h_x idx
  rw [hL, hR] at hview
  simpa using hview

/-! ## Trust audit (compile-time gate)

These commands re-audit the public result every time the file is elaborated —
if either gate fails (a smuggled axiom / `sorry`, or a foreign constant in the
trusted statement) the file stops compiling. See
`VeriTile.Meta.StatementAudit`. -/

-- (1) No `sorry`, no smuggled axiom, in the public theorem's transitive proof.
#axiomsClean float_softmax_reciprocal_refinement_view

-- (2) The headline is a *kernel-vs-kernel* refinement: its statement's project
-- surface must stay within the allowlist below — NO spec.
#stmtSurfaceSubset float_softmax_reciprocal_refinement_view ⊆
  [floatStableSoftmaxKernel, floatSoftmaxRecipKernel, ComputeKernel.eraseDType,
   ComputeRefine.General, TensorView.observe, TensorView.loaded, programTileView,
   TileIndex, BlockState, RegionName]

end FloatDTypeRefine.theorems

end VeriTile.Bench.Examples.FloatDTypeRefine
