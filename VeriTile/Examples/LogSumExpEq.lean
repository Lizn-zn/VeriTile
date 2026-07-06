/-
VeriTile.Examples.LogSumExpEq

Worked equivalence example: direct log-sum-exp kernel vs shift-trick
log-sum-exp kernel. The math identity `log_sum_exp_shift_invariant` is the
load-bearing fact; the kernel-level theorem composes operational walk-throughs
of both kernels with this identity.

Status: math identity and kernel-level refinement are closed over the typed
tile semantics.

Source Triton (`.py` reference, single-block-per-row flavour):

```python
@triton.jit
def direct_lse_kernel(x_ptr, y_ptr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    x = tl.load(x_ptr + offs)
    e = tl.exp(x)
    s = tl.sum(e, axis=0)
    y = tl.log(s)

    tl.store(y_ptr + pid, y)

@triton.jit
def stable_lse_kernel(x_ptr, y_ptr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    x = tl.load(x_ptr + offs)
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    s = tl.sum(e, axis=0)
    y = m + tl.log(s)

    tl.store(y_ptr + pid, y)
```
-/

import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.DSL
import VeriTile.Triton.Math.LogSumExp
import VeriTile.Triton.Math.Softmax
import VeriTile.Examples.Common
import VeriTile.Examples.SoftmaxEq

namespace VeriTile.Examples

open VeriTile.Triton VeriTile.Triton.TiledLogSumExp VeriTile.Triton.TiledSoftmax

/-- Direct log-sum-exp kernel: y = log(Σ exp(x)). -/
def directLSEKernel (xReg yReg : RegionName) (blockSize : Nat) : ComputeKernel := triton {
  pid  := tl.program_id(0)
  offs := pid * $(blockSize) + tl.arange(0, $(blockSize))
  x    := tl.load($(xReg) + offs)
  e    := tl.exp(x)
  s    := tl.sum(e, axis=0)
  y    := tl.log(s)
  tl.store($(yReg) + pid, y)
}

/-- Shift-trick LSE kernel: y = m + log(Σ exp(x - m)) where m = max(x). -/
def stableLSEKernel (xReg yReg : RegionName) (blockSize : Nat) : ComputeKernel := triton {
  pid  := tl.program_id(0)
  offs := pid * $(blockSize) + tl.arange(0, $(blockSize))
  x    := tl.load($(xReg) + offs)
  m    := tl.max(x, axis=0)
  e    := tl.exp(x - m)
  s    := tl.sum(e, axis=0)
  y    := m + tl.log(s)
  tl.store($(yReg) + pid, y)
}

/-! ## Kernel-level refinement -/

/-- Read region `region` at offset `basePid` (single scalar per program_id).
    Unlike softmax which writes a tile, LSE writes one cell per pid. -/
noncomputable def observeLSE
    (sf : Option BlockState) (region : RegionName) (basePid : Nat) : Option ℝ :=
  sf.map (·.readMem region basePid)

/-- **Direct LSE kernel correctness.** Single-cell observation at Y[pid]. -/
theorem direct_lse_correct
    (xReg yReg : RegionName)
    (N : Nat) (_hN : 0 < N) (s : BlockState) (xs : Fin N → ℝ)
    (_h_x : InputLoadedAt s xReg N xs) :
    observeLSE (exec (directLSEKernel xReg yReg N) s) yReg s.pid
      = some (LSE xs Bool.false 0) := by
  simp [observeLSE, exec, directLSEKernel, stepStmts, stepStmt, evalOp,
        Tile.bop, Tile.uop, Tile.reduceSum, Tile.reduceSumDrop,
        TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
        NumericDType.add, NumericDType.mul, LSE]
  repeat unfold evalOp
  simp [observeLSE, exec, directLSEKernel, stepStmts, stepStmt,
        Tile.bop, Tile.uop, Tile.reduceSum, Tile.reduceSumDrop,
        TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
        NumericDType.add, NumericDType.mul, LSE]
  unfold InputLoadedAt at _h_x
  simp_rw [_h_x]
  rfl

/-- **Stable LSE kernel correctness.** Single-cell observation; closed form
    is `m + log(Σ exp(x - m))` where `m = tileMax xs`. -/
theorem stable_lse_correct
    (xReg yReg : RegionName)
    (N : Nat) (hN : 0 < N) (s : BlockState) (xs : Fin N → ℝ)
    (_h_x : InputLoadedAt s xReg N xs) :
    observeLSE (exec (stableLSEKernel xReg yReg N) s) yReg s.pid
      = some (stableLSEWithShift xs (tileMax hN xs)) := by
  obtain ⟨n, rfl⟩ := Nat.exists_eq_succ_of_ne_zero hN.ne'
  simp [observeLSE, exec, stableLSEKernel, stepStmts, stepStmt, evalOp,
        Tile.bop, Tile.uop, Tile.reduceSum, Tile.reduceSumDrop,
        Tile.reduceMax, Tile.reduceMaxDrop,
        TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
        NumericDType.add, NumericDType.mul, NumericDType.sub,
        stableLSEWithShift, tileMax]
  repeat unfold evalOp
  simp [observeLSE, exec, stableLSEKernel, stepStmts, stepStmt,
        Tile.bop, Tile.uop, Tile.reduceSum, Tile.reduceSumDrop,
        Tile.reduceMax, Tile.reduceMaxDrop,
        TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
        NumericDType.add, NumericDType.mul, NumericDType.sub,
        stableLSEWithShift, tileMax]
  unfold InputLoadedAt at _h_x
  simp_rw [_h_x]
  rfl

/-- **Refinement: `directLSEKernel` and `stableLSEKernel` produce the same
    `Y[pid]` value.** Composes the two correctness lemmas via the math
    identity `log_sum_exp_shift_invariant`. -/
theorem log_sum_exp_refinement
    (xReg yReg : RegionName)
    (N : Nat) (hN : 0 < N) (s : BlockState) (xs : Fin N → ℝ)
    (h_x : InputLoadedAt s xReg N xs) :
    observeLSE (exec (directLSEKernel xReg yReg N) s) yReg s.pid =
    observeLSE (exec (stableLSEKernel xReg yReg N) s) yReg s.pid := by
  rw [direct_lse_correct xReg yReg N hN s xs h_x,
      stable_lse_correct xReg yReg N hN s xs h_x]
  congr 1
  unfold LSE stableLSEWithShift
  simp
  exact log_sum_exp_shift_invariant hN xs (tileMax hN xs)

/-- View-level surface for `log_sum_exp_refinement`. -/
theorem log_sum_exp_refinement_exec_view
    (xReg yReg : RegionName)
    (N : Nat) (hN : 0 < N) (s : BlockState) (xs : Fin N → ℝ)
    (h_x : TensorView.loaded s (programTileView s xReg N)
      (fun idx : TileIndex [N] => xs idx.1)) :
    TensorView.observe (exec (directLSEKernel xReg yReg N) s)
        (scalarCellView yReg s.pid) PUnit.unit =
    TensorView.observe (exec (stableLSEKernel xReg yReg N) s)
        (scalarCellView yReg s.pid) PUnit.unit := by
  have hx := inputLoadedAt_of_programTileView_loaded (s := s) (region := xReg)
    (N := N) (xs := xs) h_x
  simpa [TensorView.observe, observeTileAt, scalarCellView,
         TensorView.offset, Offset.strided, observeLSE]
    using log_sum_exp_refinement xReg yReg N hN s xs hx

end VeriTile.Examples
