/-
VeriTile.Examples.FloatDType

Worked example for VeriTile's float-facing theorem policy.

The kernel is written with explicit `tl.float32` memory annotations. The
algorithmic correctness theorem is discharged by erasing those annotations and
reusing the Real-valued VectorAdd proof.
-/

import VeriTile.Triton.Float
import VeriTile.Triton.MemoryTyping
import VeriTile.Triton.DSL
import VeriTile.Examples.VectorAdd

namespace VeriTile.Examples

open VeriTile.Triton

/-! ## Float-facing kernel -/

/-- Elementwise add with fp32-annotated input/output memory.

The surface kernel carries Triton-like dtype information, while the proof path
below erases it back to the mathematical Real kernel. -/
def floatAddKernel (xReg yReg outReg : RegionName) (blockSize : Nat) : Kernel := triton {
  pid  := tl.program_id(0)
  offs := pid * $(blockSize) + tl.arange(0, $(blockSize))
  x    := tl.load($(xReg) + offs, dtype=tl.float32)
  y    := tl.load($(yReg) + offs, dtype=tl.float32)
  out  := x + y
  tl.store($(outReg) + offs, out)
}

/-! ## Erasure bridge -/

/-- The fp32-annotated kernel erases to the existing Real VectorAdd kernel. -/
theorem float_add_erases_to_real
    (xReg yReg outReg : RegionName) (blockSize : Nat) :
    (floatAddKernel xReg yReg outReg blockSize).eraseFloat =
      addKernel xReg yReg outReg blockSize := by
  simp [floatAddKernel, addKernel, Kernel.eraseFloat,
    Stmt.eraseFloatList, Stmt.eraseFloat, Op.eraseFloat,
    eraseFloatDType, NumericDType.eraseFloat]

/-- Generic theorem bridge: any Real proof about `addKernel` can be exposed as
a float-facing theorem for `floatAddKernel` through erasure. -/
theorem float_add_correct_bridge
    (xReg yReg outReg : RegionName) (blockSize : Nat)
    {post : BlockState → BlockState → Prop}
    (hreal : Kernel.Correct (addKernel xReg yReg outReg blockSize) post) :
    Kernel.AlgorithmCorrect (floatAddKernel xReg yReg outReg blockSize) post :=
  Kernel.algorithmCorrect_of_erase_eq
    (float_add_erases_to_real xReg yReg outReg blockSize) hreal

/-- The fp32-annotated kernel algorithmically refines the existing Real kernel:
after erasure, both executions produce the same final state. -/
theorem float_add_refines_real_add
    (xReg yReg outReg : RegionName) (blockSize : Nat) :
    Kernel.AlgorithmRefine
      (floatAddKernel xReg yReg outReg blockSize)
      (addKernel xReg yReg outReg blockSize)
      (fun _ floatFinal realFinal => floatFinal = realFinal) := by
  refine Kernel.algorithmRefine_of_erase_eq
    (float_add_erases_to_real xReg yReg outReg blockSize) ?_ ?_
  · simp [addKernel, Kernel.eraseFloat, Stmt.eraseFloatList, Stmt.eraseFloat,
      Op.eraseFloat, eraseFloatDType, NumericDType.eraseFloat]
  · intro s lhs' rhs' hl hr
    exact (Option.some.inj (hl.trans hr.symm))

/-! ## Reused correctness theorem -/

/-- Float-facing VectorAdd correctness view.

The theorem statement starts from the fp32-annotated kernel, but the formal
algorithmic proof runs the erased Real semantics: state float, prove Real. -/
theorem float_add_kernel_correct_view
    (xReg yReg outReg : RegionName)
    (blockSize : Nat) (hBlockSize : 0 < blockSize)
    (s : BlockState) (xs ys : Fin blockSize → ℝ)
    (h_x : TensorView.loaded s (programTileView s xReg blockSize)
      (fun idx : TileIndex [blockSize] => xs idx.1))
    (h_y : TensorView.loaded s (programTileView s yReg blockSize)
      (fun idx : TileIndex [blockSize] => ys idx.1)) :
    ∀ idx : TileIndex [blockSize],
      TensorView.observe
          (exec (floatAddKernel xReg yReg outReg blockSize).eraseFloat s)
          (programTileView s outReg blockSize) idx
        = some (addSpec xs ys idx.1) := by
  have hreal := add_kernel_correct_view xReg yReg outReg blockSize hBlockSize
    s xs ys h_x h_y
  simpa [float_add_erases_to_real] using hreal

/-! ## Lightweight memory typing -/

/-- The fp32-annotated kernel satisfies the lightweight region typing contract
when all participating buffers are declared fp32. -/
theorem float_add_respects_fp32_regions
    (xReg yReg outReg : RegionName) (blockSize : Nat) :
    (floatAddKernel xReg yReg outReg blockSize).RespectsRegionTyping
      (fun _ => TileDType.fp32) := by
  simp [floatAddKernel, Kernel.RespectsRegionTyping, StmtList.RespectsRegionTyping,
    Stmt.RespectsRegionTyping, Op.RespectsRegionTyping, FloatDType.dtype]

end VeriTile.Examples
