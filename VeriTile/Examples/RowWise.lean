/-
VeriTile.Examples.RowWise

Worked correctness examples for row-wise reductions over a row-major 2D
matrix. Each `program_id` processes one row: gathers `blockSize` consecutive
cells starting at `xReg + row * nCol`, reduces the tile, and stores one scalar
result at `yReg + row`.

The examples use 1D tiles and row-stride arithmetic (`row * nCol + cols`).
Real Triton code normally adds a tail mask (`cols < n_cols`) when the row
length is not exactly covered; these aligned examples assume the row segment
is loaded by `InputRowLoadedAt`.
-/

import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import VeriTile.Core
import VeriTile.Semantics
import VeriTile.Float
import VeriTile.Frontend.Triton.DSL
import VeriTile.Math.Reduction
import VeriTile.Examples.Common

namespace VeriTile.Examples

open VeriTile

/-! ## Source Triton shape

```python
@triton.jit
def row_wise_sum(X, Y, n_cols: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    row    = tl.program_id(0)
    cols   = tl.arange(0, BLOCK_SIZE)
    values = tl.load(X + row * n_cols + cols)
    result = tl.sum(values, axis=0)
    tl.store(Y + row, result)

@triton.jit
def row_wise_max(X, Y, n_cols: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    row    = tl.program_id(0)
    cols   = tl.arange(0, BLOCK_SIZE)
    values = tl.load(X + row * n_cols + cols)
    result = tl.max(values, axis=0)
    tl.store(Y + row, result)
```
-/

/-! ## Embedded Triton ASTs -/

/-- Row-wise sum over a row-major 2D matrix.

Per `program_id`: gather `blockSize` cells from row `pid` of `xReg` (row
stride `nCol`), sum them, and scatter the scalar to `yReg[pid]`. -/
def rowWiseSumKernel (xReg yReg : RegionName) (nCol blockSize : Nat) : ComputeKernel :=
  triton {
    row    := tl.program_id(0)
    cols   := tl.arange(0, $(blockSize))
    values := tl.load($(xReg) + row * $(nCol) + cols)
    result := tl.sum(values, axis=0)
    tl.store($(yReg) + row, result)
  }

/-- Row-wise max over a row-major 2D matrix.

Per `program_id`: gather `blockSize` cells from row `pid` of `xReg` (row
stride `nCol`), reduce by max, and scatter the scalar to `yReg[pid]`. -/
def rowWiseMaxKernel (xReg yReg : RegionName) (nCol blockSize : Nat) : ComputeKernel :=
  triton {
    row    := tl.program_id(0)
    cols   := tl.arange(0, $(blockSize))
    values := tl.load($(xReg) + row * $(nCol) + cols)
    result := tl.max(values, axis=0)
    tl.store($(yReg) + row, result)
  }

/-! ## Math denotations -/

/-- Mathematical row-wise sum. Thin alias for `TiledReduction.tileSum`. -/
noncomputable def rowWiseSumSpec {N : Nat} (xs : Fin N → ℝ) : ℝ :=
  TiledReduction.tileSum xs

/-- Mathematical row-wise max. Thin alias for `TiledReduction.tileMax`. -/
noncomputable def rowWiseMaxSpec {N : Nat} (h : 0 < N) (xs : Fin N → ℝ) : ℝ :=
  TiledReduction.tileMax h xs

/-! ## Kernel correctness -/

/-- **Row-wise sum kernel correctness.**

After running `rowWiseSumKernel xReg yReg nCol blockSize` on a state where
row `s.pid` of `xReg` holds the tile `xs`, the cell `yReg[s.pid]` equals
`∑ xs`. -/
theorem rowWiseSum_correct
    (xReg yReg : RegionName) (nCol blockSize : Nat)
    (s : BlockState) (xs : Fin blockSize → ℝ)
    (h_x : InputRowLoadedAt s xReg nCol blockSize xs) :
    observeRowAt (exec (rowWiseSumKernel xReg yReg nCol blockSize) s) yReg s.pid
      = some (rowWiseSumSpec xs) := by
  simp [observeRowAt, exec, rowWiseSumKernel, stepStmts, stepStmt, evalOp,
        Tile.bop, Tile.reduceSum, Tile.reduceSumDrop,
        TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
        NumericDType.mul, NumericDType.add,
        rowWiseSumSpec, TiledReduction.tileSum]
  unfold InputRowLoadedAt at h_x
  simp_rw [h_x]
  apply Exists.intro
  constructor
  · unfold evalOp
    simp [Tile.reduceSum, Tile.reduceSumDrop,
      TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex]
    rfl
  · simp [BlockState.writeMem_readMem]
    rfl

/-- View-level surface for `rowWiseSum_correct`. -/
theorem rowWiseSum_correct_exec_view
    (xReg yReg : RegionName) (nCol blockSize : Nat)
    (s : BlockState) (xs : Fin blockSize → ℝ)
    (h_x : TensorView.loaded s (rowTileView s xReg nCol blockSize)
      (fun idx : TileIndex [blockSize] => xs idx.1)) :
    TensorView.observe (exec (rowWiseSumKernel xReg yReg nCol blockSize) s)
        (scalarCellView yReg s.pid) PUnit.unit
      = some (rowWiseSumSpec xs) := by
  have hx := inputRowLoadedAt_of_rowTileView_loaded (s := s) (region := xReg)
    (rowStride := nCol) (blockSize := blockSize) (xs := xs) h_x
  simpa [TensorView.observe, observeTileAt, scalarCellView,
         TensorView.offset, Offset.strided, observeRowAt, BlockState.readMem]
    using rowWiseSum_correct xReg yReg nCol blockSize s xs hx

/-- Compute-facing view-level surface for `rowWiseSum_correct`. -/
theorem rowWiseSum_correct_view
    (xReg yReg : RegionName) (nCol blockSize : Nat)
    (s : BlockState) (xs : Fin blockSize → ℝ)
    (h_x : TensorView.loaded s (rowTileView s xReg nCol blockSize)
      (fun idx : TileIndex [blockSize] => xs idx.1)) :
    ComputeCorrect.Realizes
      (kernel := rowWiseSumKernel xReg yReg nCol blockSize)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.scalar yReg s.pid)
      (expected := fun _ : PUnit => rowWiseSumSpec xs) := by
  apply ComputeKernel.computeCorrect_of_toAlgKernel rfl
  intro s0 s' hExec hs0
  subst s0
  have hview := rowWiseSum_correct_exec_view xReg yReg nCol blockSize s xs h_x
  rw [hExec] at hview
  intro _
  simpa [TensorView.observe, observeTileAt, scalarCellView, TensorView.offset,
    Offset.strided] using hview

/-- **Row-wise max kernel correctness.**

After running `rowWiseMaxKernel xReg yReg nCol blockSize` on a state where
row `s.pid` of `xReg` holds the non-empty tile `xs`, the cell `yReg[s.pid]`
equals `max xs`. -/
theorem rowWiseMax_correct
    (xReg yReg : RegionName) (nCol blockSize : Nat) (hN : 0 < blockSize)
    (s : BlockState) (xs : Fin blockSize → ℝ)
    (h_x : InputRowLoadedAt s xReg nCol blockSize xs) :
    observeRowAt (exec (rowWiseMaxKernel xReg yReg nCol blockSize) s) yReg s.pid
      = some (rowWiseMaxSpec hN xs) := by
  obtain ⟨n, rfl⟩ := Nat.exists_eq_succ_of_ne_zero hN.ne'
  simp [observeRowAt, exec, rowWiseMaxKernel, stepStmts, stepStmt, evalOp,
        Tile.bop, Tile.reduceMax, Tile.reduceMaxDrop,
        TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
        NumericDType.mul, NumericDType.add,
        rowWiseMaxSpec, TiledReduction.tileMax]
  unfold InputRowLoadedAt at h_x
  simp_rw [h_x]
  apply Exists.intro
  constructor
  · unfold evalOp
    simp [Tile.reduceMax, Tile.reduceMaxDrop,
      TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex]
    rfl
  · simp [BlockState.writeMem_readMem]
    rfl

/-- View-level surface for `rowWiseMax_correct`. -/
theorem rowWiseMax_correct_exec_view
    (xReg yReg : RegionName) (nCol blockSize : Nat) (hN : 0 < blockSize)
    (s : BlockState) (xs : Fin blockSize → ℝ)
    (h_x : TensorView.loaded s (rowTileView s xReg nCol blockSize)
      (fun idx : TileIndex [blockSize] => xs idx.1)) :
    TensorView.observe (exec (rowWiseMaxKernel xReg yReg nCol blockSize) s)
        (scalarCellView yReg s.pid) PUnit.unit
      = some (rowWiseMaxSpec hN xs) := by
  have hx := inputRowLoadedAt_of_rowTileView_loaded (s := s) (region := xReg)
    (rowStride := nCol) (blockSize := blockSize) (xs := xs) h_x
  simpa [TensorView.observe, observeTileAt, scalarCellView,
         TensorView.offset, Offset.strided, observeRowAt, BlockState.readMem]
    using rowWiseMax_correct xReg yReg nCol blockSize hN s xs hx

/-- Compute-facing view-level surface for `rowWiseMax_correct`. -/
theorem rowWiseMax_correct_view
    (xReg yReg : RegionName) (nCol blockSize : Nat) (hN : 0 < blockSize)
    (s : BlockState) (xs : Fin blockSize → ℝ)
    (h_x : TensorView.loaded s (rowTileView s xReg nCol blockSize)
      (fun idx : TileIndex [blockSize] => xs idx.1)) :
    ComputeCorrect.Realizes
      (kernel := rowWiseMaxKernel xReg yReg nCol blockSize)
      (initialState := s)
      (write := ComputeCorrect.WriteMap.scalar yReg s.pid)
      (expected := fun _ : PUnit => rowWiseMaxSpec hN xs) := by
  apply ComputeKernel.computeCorrect_of_toAlgKernel rfl
  intro s0 s' hExec hs0
  subst s0
  have hview := rowWiseMax_correct_exec_view xReg yReg nCol blockSize hN s xs h_x
  rw [hExec] at hview
  intro _
  simpa [TensorView.observe, observeTileAt, scalarCellView, TensorView.offset,
    Offset.strided] using hview

end VeriTile.Examples
