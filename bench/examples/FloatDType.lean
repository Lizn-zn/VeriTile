/-
VeriTile.Examples.FloatDType

Worked example for VeriTile's float-facing theorem policy.

The kernel is written with explicit `tl.float32` memory annotations. The
algorithmic correctness theorem is discharged by erasing those annotations and
reusing the Real-valued VectorAdd proof.
-/

import VeriTile.Triton.Float
import VeriTile.Triton.Memory.Typing
import VeriTile.Triton.DSL
import VeriTile.Examples.VectorAdd
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

/-! ## Float-facing kernel -/

/-- Elementwise add with fp32-annotated input/output memory.

The surface kernel carries Triton-like dtype information, while the proof path
below erases it back to the mathematical Real kernel. -/
def floatAddKernel (xReg yReg outReg : RegionName) (blockSize : Nat) : ComputeKernel := triton {
  pid  := tl.program_id(0)
  offs := pid * $(blockSize) + tl.arange(0, $(blockSize))
  x    := tl.load($(xReg) + offs, dtype=tl.float32)
  y    := tl.load($(yReg) + offs, dtype=tl.float32)
  out  := x + y
  tl.store($(outReg) + offs, out)
}

/-! ## Erasure bridge -/

/-- The fp32-annotated kernel erases to the existing Real VectorAdd kernel,
at the algorithm-projection level: both sides project (via `toAlgKernel`) to
the same plain algorithm kernel. -/
theorem float_add_erases_to_real
    (xReg yReg outReg : RegionName) (blockSize : Nat) :
    (floatAddKernel xReg yReg outReg blockSize).eraseDType.toAlgKernel =
      (addKernel xReg yReg outReg blockSize).toAlgKernel := by
  simp [floatAddKernel, addKernel, ComputeKernel.eraseDType,
    ComputeStmt.toAlgorithm?, ComputeStmt.listToAlgorithm?,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, ComputeDType.eraseDType,
    Kernel.eraseDType, Stmt.eraseDTypeList, Stmt.eraseDType,
    Op.eraseDType, VeriTile.Triton.eraseDType, NumericDType.eraseDType]

/-! ## Reused correctness theorem -/

/-- Float-facing VectorAdd correctness view.

The theorem statement starts from the fp32-annotated kernel, but the formal
algorithmic proof runs the erased Real semantics: state float, prove Real. -/
theorem float_add_kernel_correct_exec_view
    (xReg yReg outReg : RegionName)
    (blockSize : Nat) (hBlockSize : 0 < blockSize)
    (s : BlockState) (xs ys : Fin blockSize → ℝ)
    (h_x : TensorView.loaded s (programTileView s xReg blockSize)
      (fun idx : TileIndex [blockSize] => xs idx.1))
    (h_y : TensorView.loaded s (programTileView s yReg blockSize)
      (fun idx : TileIndex [blockSize] => ys idx.1)) :
    ∀ idx : TileIndex [blockSize],
      TensorView.observe
          (exec (floatAddKernel xReg yReg outReg blockSize).eraseDType s)
          (programTileView s outReg blockSize) idx
        = some (addSpec xs ys idx.1) := by
  intro idx
  rw [float_add_erases_to_real xReg yReg outReg blockSize]
  exact add_kernel_correct_exec_view xReg yReg outReg blockSize hBlockSize
    s xs ys h_x h_y idx

/-- Compute-facing float VectorAdd correctness view, projected to the erased
algorithm kernel. -/
theorem float_add_kernel_correct_view
    (xReg yReg outReg : RegionName)
    (blockSize : Nat) (hBlockSize : 0 < blockSize)
    (s : BlockState) (xs ys : Fin blockSize → ℝ)
    (h_x : TensorView.loaded s (programTileView s xReg blockSize)
      (fun idx : TileIndex [blockSize] => xs idx.1))
    (h_y : TensorView.loaded s (programTileView s yReg blockSize)
      (fun idx : TileIndex [blockSize] => ys idx.1)) :
    ComputeCorrect.General
      (((floatAddKernel xReg yReg outReg blockSize).eraseDType))
      (fun s0 s' =>
        s0 = s →
        ∀ idx : TileIndex [blockSize],
          TensorView.observe (some s')
              (programTileView s outReg blockSize) idx
            = some (addSpec xs ys idx.1)) := by
  apply ComputeKernel.computeCorrect_of_toAlgKernel
    (by simp [ComputeKernel.eraseDType])
  intro s0 s' hExec hs0
  subst s0
  intro idx
  have hview := float_add_kernel_correct_exec_view xReg yReg outReg blockSize hBlockSize
    s xs ys h_x h_y idx
  rw [hExec] at hview
  simpa using hview

/-! ## Lightweight memory typing -/

/-- The original claim here — `Kernel.RespectsRegionTyping (fun _ => .fp32)
(floatAddKernel …)` — is **false** under today's definitions and is therefore
recorded as a proved negation. `Kernel.RespectsRegionTyping` receives the
kernel through the `ComputeKernel → AlgKernel` coercion (`toAlgKernel`), and
the compute→algorithm projection (`ComputeOp.toAlgorithm?`) rewrites every
`tl.float32` load to `Op.load ComputeDType.fp32.eraseDType = Op.load .real`
before the typing contract ever sees it. The contract's load/store cases then
demand `Γ region = .real`, i.e. `TileDType.fp32 = TileDType.real`, which
reduces to `False`. -/
theorem float_add_not_respects_fp32_regions
    (xReg yReg outReg : RegionName) (blockSize : Nat) :
    ¬ Kernel.RespectsRegionTyping (fun _ => .fp32)
      (floatAddKernel xReg yReg outReg blockSize) := by
  simp [floatAddKernel, ComputeKernel.toAlgKernel,
    ComputeStmt.toAlgorithm?, ComputeStmt.listToAlgorithm?,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, ComputeDType.eraseDType,
    Kernel.RespectsRegionTyping, StmtList.RespectsRegionTyping,
    Stmt.RespectsRegionTyping, Op.RespectsRegionTyping,
    MemAccess.RespectsRegionTyping, MaskOpt.RespectsRegionTyping]

/-- The true region-typing contract on today's projected surface: after the
compute→algorithm projection the fp32 annotations are erased, so the kernel
respects the all-`.real` region typing. -/
theorem float_add_respects_real_regions
    (xReg yReg outReg : RegionName) (blockSize : Nat) :
    Kernel.RespectsRegionTyping (fun _ => .real)
      (floatAddKernel xReg yReg outReg blockSize) := by
  simp [floatAddKernel, ComputeKernel.toAlgKernel,
    ComputeStmt.toAlgorithm?, ComputeStmt.listToAlgorithm?,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, ComputeDType.eraseDType,
    Kernel.RespectsRegionTyping, StmtList.RespectsRegionTyping,
    Stmt.RespectsRegionTyping, Op.RespectsRegionTyping,
    MemAccess.RespectsRegionTyping, MaskOpt.RespectsRegionTyping]

/-! ## Float-facing rewrite refinement -/

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
