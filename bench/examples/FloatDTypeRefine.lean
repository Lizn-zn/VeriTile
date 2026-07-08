/-
# `VeriTile.Examples.FloatDTypeRefine` — dtype erasure as the float-facing
# *refinement* policy

Worked example for VeriTile's float-facing theorem policy on the
**refinement** surface (`ComputeRefine`): two kernels carry explicit
`tl.float32` memory annotations, and the algorithmic equivalence theorem is
discharged by **erasing** those annotations to the Real channel and reusing
the Real-valued refinement (here the softmax reciprocal rewrite). This is the
erased/ideal pathway — the complement of the rounding-model pathway
(`ComputeRefine.*R`) demonstrated in `bench/examples/Swiglu`.

The correctness counterpart of this policy (one annotated kernel shown to
realize a spec through erasure) lives in
`bench/examples/FloatDTypeCorrect.lean`.

## Kernels & theorems

Kernels: `floatStableSoftmaxKernel`, `floatSoftmaxRecipKernel`.

* `float_stable_softmax_erases_to_real` / `float_softmax_recip_erases_to_real`
  — each annotated kernel's projection erases to the Real-typed kernel (the
  reuse bridge).
* `float_softmax_reciprocal_refinement_view` / `…_exec_view` — the erased
  equivalence statement (`ComputeRefine.General`): the per-element-divide and
  reciprocal-form softmaxes agree cell-by-cell.
-/

import VeriTile.Triton
import VeriTile.Examples.SoftmaxReciprocal

namespace VeriTile.Examples

open VeriTile.Triton

/-- Local reduction helper: the algorithm-projection traversals
(`ComputeStmt.toAlgorithm?` and friends) are written in `do` notation over
`Except`, so their equation lemmas leave `Except.ok _ >>= f` redexes behind.
This discharges them for `simp`. -/
@[simp] private theorem except_ok_bind {α β ε : Type _} (a : α)
    (f : α → Except ε β) :
    (Except.ok a : Except ε α) >>= f = f a := rfl

/-! ## Float-facing kernels -/

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

/-! ## Erasure bridge -/

/-- The fp32 per-element-divide softmax erases to the existing Real kernel,
at the algorithm-projection level. -/
theorem float_stable_softmax_erases_to_real
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
theorem float_softmax_recip_erases_to_real
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

/-! ## Reused refinement theorem -/

/-- Float-facing rewrite refinement: the fp32 per-element-divide softmax and
the fp32 reciprocal-form softmax are algorithmically equivalent after erasure.

This is the float theorem policy applied to a real optimization: state the
dtype-annotated rewrite, prove it through the existing Real refinement. -/
theorem float_softmax_reciprocal_refinement_exec_view
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

/-- Compute-facing surface for the fp32 reciprocal-form softmax rewrite,
projected to the erased algorithm kernels. -/
theorem float_softmax_reciprocal_refinement_view
    (xReg yReg : RegionName)
    (N : Nat) (hN : 0 < N) (s : BlockState) (xs : Fin N → ℝ)
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

end VeriTile.Examples
