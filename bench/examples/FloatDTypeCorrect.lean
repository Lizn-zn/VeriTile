/-
# `VeriTile.Examples.FloatDTypeCorrect` — dtype erasure as the float-facing
# *correctness* policy

Worked example for VeriTile's float-facing theorem policy on the
**correctness** surface (`ComputeCorrect`): a kernel carries explicit
`tl.float32` memory annotations, and the algorithmic correctness theorem is
discharged by **erasing** those annotations to the Real channel and reusing
the Real-valued proof (here VectorAdd's). This is the erased/ideal pathway —
the complement of the rounding-model pathway (`ComputeRefine.*R`) demonstrated
in `bench/examples/Swiglu`.

The refinement counterpart of this policy (two annotated kernels shown
algorithmically equivalent through erasure) lives in
`bench/examples/FloatDTypeRefine.lean`.

## Kernel & theorems

Kernel: `floatAddKernel`.

* `float_add_erases_to_real` — the annotated kernel's projection erases to the
  Real-typed kernel (the reuse bridge).
* `float_add_kernel_correct_view` / `…_exec_view` — the erased correctness
  statement (`ComputeCorrect.General`).
* `float_add_respects_real_regions` and the honest negation
  `float_add_not_respects_fp32_regions` — the true region-typing contract holds
  on the Real channel, and the naïve fp32-region claim is **provably false**
  (the projection erases fp32 loads to Real before the typing would bite).
-/

import VeriTile.Triton
import VeriTile.Examples.VectorAdd

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

end VeriTile.Examples
