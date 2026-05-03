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
import VeriTile.Examples.SoftmaxReciprocal

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

/-! ## Float-facing rewrite refinement -/

/-- Stable softmax with explicit fp32 input/output annotations and per-element
division `y = e / s`. The reductions still run in the Real abstraction after
the input load is cast to `tl.float64`. -/
def floatStableSoftmaxKernel (xReg yReg : RegionName) (blockSize : Nat) : Kernel := triton {
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
def floatSoftmaxRecipKernel (xReg yReg : RegionName) (blockSize : Nat) : Kernel := triton {
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

/-- The fp32 per-element-divide softmax erases to the existing Real kernel. -/
theorem float_stable_softmax_erases_to_real
    (xReg yReg : RegionName) (blockSize : Nat) :
    (floatStableSoftmaxKernel xReg yReg blockSize).eraseFloat =
      stableSoftmaxKernel xReg yReg blockSize := by
  simp [floatStableSoftmaxKernel, stableSoftmaxKernel, Kernel.eraseFloat,
    Stmt.eraseFloatList, Stmt.eraseFloat, Op.eraseFloat,
    eraseFloatDType, NumericDType.eraseFloat]
  constructor
  · unfold Op.eraseFloat
    change (Op.reduceMax 0 Bool.false ((Op.ref TileDType.real [blockSize] "x" : Op .real [blockSize]).eraseFloat) = Op.reduceMax 0 Bool.false (Op.ref TileDType.real [blockSize] "x"))
    simp [Op.eraseFloat, eraseFloatDType]
  · unfold Op.eraseFloat
    change (Op.reduceSum 0 Bool.false ((Op.ref TileDType.real [blockSize] "e" : Op .real [blockSize]).eraseFloat) = Op.reduceSum 0 Bool.false (Op.ref TileDType.real [blockSize] "e"))
    simp [Op.eraseFloat, eraseFloatDType]

/-- The fp32 reciprocal-form softmax erases to the existing Real optimized
kernel. -/
theorem float_softmax_recip_erases_to_real
    (xReg yReg : RegionName) (blockSize : Nat) :
    (floatSoftmaxRecipKernel xReg yReg blockSize).eraseFloat =
      softmaxRecipKernel xReg yReg blockSize := by
  simp [floatSoftmaxRecipKernel, softmaxRecipKernel, Kernel.eraseFloat,
    Stmt.eraseFloatList, Stmt.eraseFloat, Op.eraseFloat,
    eraseFloatDType, NumericDType.eraseFloat]
  constructor
  · unfold Op.eraseFloat
    change (Op.reduceMax 0 Bool.false ((Op.ref TileDType.real [blockSize] "x" : Op .real [blockSize]).eraseFloat) = Op.reduceMax 0 Bool.false (Op.ref TileDType.real [blockSize] "x"))
    simp [Op.eraseFloat, eraseFloatDType]
  · unfold Op.eraseFloat
    change (Op.reduceSum 0 Bool.false ((Op.ref TileDType.real [blockSize] "e" : Op .real [blockSize]).eraseFloat) = Op.reduceSum 0 Bool.false (Op.ref TileDType.real [blockSize] "e"))
    simp [Op.eraseFloat, eraseFloatDType]

/-- Float-facing rewrite refinement: the fp32 per-element-divide softmax and
the fp32 reciprocal-form softmax are algorithmically equivalent after erasure.

This is the float theorem policy applied to a real optimization: state the
dtype-annotated rewrite, prove it through the existing Real refinement. -/
theorem float_softmax_reciprocal_algorithm_refine_view
    (xReg yReg : RegionName)
    (N : Nat) (hN : 0 < N) (s : BlockState) (xs : Fin N → ℝ)
    (h_x : TensorView.loaded s (programTileView s xReg N)
      (fun idx : TileIndex [N] => xs idx.1)) :
    Kernel.AlgorithmRefine
      (floatStableSoftmaxKernel xReg yReg N)
      (floatSoftmaxRecipKernel xReg yReg N)
      (fun init divFinal recipFinal =>
        init = s →
        ∀ idx : TileIndex [N],
          TensorView.observe (some divFinal) (programTileView s yReg N) idx =
          TensorView.observe (some recipFinal) (programTileView s yReg N) idx) := by
  refine Kernel.algorithmRefine_of_erase_eq
    (float_stable_softmax_erases_to_real xReg yReg N)
    (float_softmax_recip_erases_to_real xReg yReg N) ?_
  intro init divFinal recipFinal hDiv hRecip hInit idx
  subst init
  have h := softmax_reciprocal_refinement_view xReg yReg N hN s xs h_x idx
  simpa [hDiv, hRecip] using h

end VeriTile.Examples
