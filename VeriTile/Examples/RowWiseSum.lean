/-
VeriTile.Examples.RowWiseSum

Worked correctness example: row-wise sum over a row-major 2D matrix.

Each `program_id` processes one row: gathers `blockSize` consecutive cells
starting at `xReg + row * nCol`, sums them, stores the scalar result at
`yReg + row`.

This is the simplest *2D-style* gather kernel expressible without 2D tile
infra (issue #4) — the offsets are computed by 1D arithmetic
`row * nCol + cols` but the result is still a 1D tile, so we only need
the existing `Value.tile` carrier and the existing `Op.reduceSum`. Once
2D tiles land we will revisit this kernel as a sanity check that the new
infra is backward-compatible.

Structure (mirroring `SoftmaxEq.lean`):

  (a) Source `.py` Triton (commented, for reference)
  (b) Embedded Triton AST via `triton { ... }` macro
  (c) Math denotation and helpers
  (d) Kernel correctness against `rowWiseSumSpec`

`rowWiseSum_correct` is closed: the proof reduces `exec` via `simp` on
the operational semantics and substitutes the loaded input cells via
`InputRowLoadedAt`, with no math content (the spec is just `∑ xs`).
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
# row_wise_sum.py
@triton.jit
def row_wise_sum(X, Y, n_cols: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    row    = tl.program_id(0)
    cols   = tl.arange(0, BLOCK_SIZE)
    values = tl.load(X + row * n_cols + cols)   # blockSize ≤ n_cols
    result = tl.sum(values, axis=0)
    tl.store(Y + row, result)
```

Note on the surface form: real Triton code adds `mask = cols < n_cols` for
the tail block when `BLOCK_SIZE` does not divide `n_cols`. Masking is
issue #3 and is out of scope here; the correctness theorem below assumes
`blockSize ≤ n_cols` *and* (for offset alignment with `InputRowLoadedAt`)
that the matrix has an exact `nCol`-element row stride. -/

/-! ## (b) Embedded Triton AST -/

/-- Row-wise sum over a row-major 2D matrix.

Per `program_id`: gather `blockSize` cells from row `pid` of `xReg` (row
stride `nCol`), sum them, scatter the scalar to `yReg[pid]`. -/
def rowWiseSumKernel (xReg yReg : RegionName) (nCol blockSize : Nat) : Kernel :=
  triton {
    row    := tl.program_id(0)
    cols   := tl.arange(0, $(blockSize))
    values := tl.load($(xReg) + row * $(nCol) + cols)
    result := tl.sum(values)
    tl.store($(yReg) + row, result)
  }

/-! ## (c) Math denotation and helpers -/

/-- Mathematical row-wise sum: just the `Finset.sum` over the tile. -/
noncomputable def rowWiseSumSpec {N : Nat} (xs : Fin N → ℝ) : ℝ :=
  ∑ i, xs i

/-! ## (d) Kernel correctness

Pattern: same ARM-in-Lean style as `SoftmaxEq.lean`, but the output is a
single scalar per block (not a tile), so the readback step is a direct
`writeMem`/`readMem` step rather than a `scatter_readback` over a `foldl`. -/

/-- **Row-wise sum kernel correctness.**

After running `rowWiseSumKernel xReg yReg nCol blockSize` on a state where
row `s.pid` of `xReg` (rows striding by `nCol`) holds the tile `xs`, the
cell `yReg[s.pid]` equals `∑ xs`.

Proof structure:

1. `simp` reduces the 5-statement body via the operational semantics.
2. The store offset is *scalar* (`row` evaluates to `scalarNat s.pid`) and
   the value is *scalar* (`result` evaluates to `scalar (∑ ...)`); the
   write lands in the `writeMem` branch of `Stmt.store`, so the resulting
   state's `mem yReg` at offset `s.pid` reads back the value directly —
   no `scatter_readback` over a tile-scatter `foldl` is needed.
3. `simp_rw [h_x]` substitutes the loaded input cells with `xs`.

Math content here: zero — the spec is just `∑ xs`. -/
theorem rowWiseSum_correct
    (xReg yReg : RegionName) (nCol blockSize : Nat)
    (s : BlockState) (xs : Fin blockSize → ℝ)
    (h_x : InputRowLoadedAt s xReg nCol blockSize xs) :
    observeRowAt (exec (rowWiseSumKernel xReg yReg nCol blockSize) s) yReg s.pid
      = some (rowWiseSumSpec xs) := by
  -- The 5-statement body reduces via `simp` on the operational semantics.
  -- The final `Stmt.store yReg (Op.ref "row") (Op.ref "result")` is a
  -- *scalar* store (offset is `scalarNat s.pid`, value is `scalar (∑ ...)`),
  -- so it lands in the `writeMem` branch — readback is direct, no
  -- `scatter_readback` over a `foldl` needed.
  simp [observeRowAt, exec, rowWiseSumKernel, stepStmts, stepStmt, evalOp,
        Value.bop, Value.reduceSum, BlockState.setReg, BlockState.readMem,
        BlockState.writeMem, rowWiseSumSpec]
  -- Substitute the loaded input cells with `xs` via `InputRowLoadedAt`.
  unfold InputRowLoadedAt at h_x
  simp_rw [h_x]

end VeriTile.Examples
