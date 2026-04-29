/-
VeriTile.Examples.RowWiseMax

Worked correctness example: row-wise max over a row-major 2D matrix.

Each `program_id` processes one row: gathers `blockSize` consecutive cells
starting at `xReg + row * nCol`, takes the max, stores the scalar result
at `yReg + row`. Companion to `RowWiseSum.lean` — same skeleton, with
`tl.sum` swapped for `tl.max`.

The proof is closed: it follows the same `simp` + `simp_rw [h_x]`
pattern as `rowWiseSum_correct`, with one extra prelude step
(`obtain ⟨n, rfl⟩ := Nat.exists_eq_succ_of_ne_zero hN.ne'`) so
`Tile.reduceMax` reduces — `Tile.reduceMax` is only defined on
`tile (n+1)` because `Finset.sup'` requires a non-empty index set.
-/

import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.DSL
import VeriTile.Examples.Common

namespace VeriTile.Examples

open VeriTile.Triton

/-! ## (a) Source Triton (`.py`) — for reference only

```python
# row_wise_max.py
@triton.jit
def row_wise_max(X, Y, n_cols: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    row    = tl.program_id(0)
    cols   = tl.arange(0, BLOCK_SIZE)
    values = tl.load(X + row * n_cols + cols)   # blockSize ≤ n_cols
    result = tl.max(values, axis=0)
    tl.store(Y + row, result)
```

Same masking caveat as `RowWiseSum`: real Triton adds
`mask = cols < n_cols` for tail blocks; masking is issue #3 and out of
scope here. The correctness theorem assumes `blockSize ≤ nCol` and (for
offset alignment with `InputRowLoadedAt`) an exact `nCol`-element row
stride. -/

/-! ## (b) Embedded Triton AST -/

/-- Row-wise max over a row-major 2D matrix.

Per `program_id`: gather `blockSize` cells from row `pid` of `xReg`
(row stride `nCol`), reduce by max, scatter the scalar to `yReg[pid]`. -/
def rowWiseMaxKernel (xReg yReg : RegionName) (nCol blockSize : Nat) : Kernel :=
  triton {
    row    := tl.program_id(0)
    cols   := tl.arange(0, $(blockSize))
    values := tl.load($(xReg) + row * $(nCol) + cols)
    result := tl.max(values)
    tl.store($(yReg) + row, result)
  }

/-! ## (c) Math denotation -/

/-- Mathematical row-wise max via Mathlib's `Finset.sup'` (needs the tile
    to be non-empty, hence the `0 < N` hypothesis). The dependent pattern
    match on `N` mirrors `Tile.reduceMax`'s shape and `tileMax` from
    `SoftmaxEq.lean`, so the closed-form output of the kernel reduces
    cleanly to this spec under `simp`. -/
noncomputable def rowWiseMaxSpec {N : Nat} (h : 0 < N) (xs : Fin N → ℝ) : ℝ :=
  match N, h, xs with
  | _ + 1, _, xs => Finset.univ.sup' Finset.univ_nonempty xs

/-! ## (d) Kernel correctness

Mirrors `rowWiseSum_correct` exactly, with one extra prelude step to
case-split `blockSize = n + 1` so `Tile.reduceMax` (defined only on
non-empty tiles) reduces. The store is still scalar — single
`writeMem`, no `scatter_readback` over a `foldl`. -/

/-- **Row-wise max kernel correctness.**

After running `rowWiseMaxKernel xReg yReg nCol blockSize` on a state
where row `s.pid` of `xReg` (rows striding by `nCol`) holds the tile
`xs`, the cell `yReg[s.pid]` equals `max xs`.

Proof structure:

1. `obtain ⟨n, rfl⟩ := Nat.exists_eq_succ_of_ne_zero hN.ne'` to expose
   `blockSize = n + 1`, so `Tile.reduceMax` matches.
2. `simp` reduces the 5-statement body via the operational semantics.
3. `simp_rw [h_x]` substitutes the loaded input cells with `xs`.

Math content here: zero — the spec is `Finset.sup' xs`. -/
theorem rowWiseMax_correct
    (xReg yReg : RegionName) (nCol blockSize : Nat) (hN : 0 < blockSize)
    (s : BlockState) (xs : Fin blockSize → ℝ)
    (h_x : InputRowLoadedAt s xReg nCol blockSize xs) :
    observeRowAt (exec (rowWiseMaxKernel xReg yReg nCol blockSize) s) yReg s.pid
      = some (rowWiseMaxSpec hN xs) := by
  obtain ⟨n, rfl⟩ := Nat.exists_eq_succ_of_ne_zero hN.ne'
  simp [observeRowAt, exec, rowWiseMaxKernel, stepStmts, stepStmt, evalOp,
        Tile.bop, Tile.reduceMax, NumericDType.mul, NumericDType.add,
        BlockState.setReg, BlockState.readMem, BlockState.writeMem,
        rowWiseMaxSpec]
  simp [Broadcast.rightIndex]
  unfold InputRowLoadedAt at h_x
  simp_rw [h_x]

end VeriTile.Examples
