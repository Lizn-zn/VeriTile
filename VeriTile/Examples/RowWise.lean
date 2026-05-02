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
import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.DSL
import VeriTile.Examples.Common

namespace VeriTile.Examples

open VeriTile.Triton

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
def rowWiseSumKernel (xReg yReg : RegionName) (nCol blockSize : Nat) : Kernel :=
  triton {
    row    := tl.program_id(0)
    cols   := tl.arange(0, $(blockSize))
    values := tl.load(tl.ptr($(xReg)) + row * $(nCol) + cols)
    result := tl.sum(values, axis=0)
    tl.store(tl.ptr($(yReg)) + row, result)
  }

/-- Row-wise max over a row-major 2D matrix.

Per `program_id`: gather `blockSize` cells from row `pid` of `xReg` (row
stride `nCol`), reduce by max, and scatter the scalar to `yReg[pid]`. -/
def rowWiseMaxKernel (xReg yReg : RegionName) (nCol blockSize : Nat) : Kernel :=
  triton {
    row    := tl.program_id(0)
    cols   := tl.arange(0, $(blockSize))
    values := tl.load(tl.ptr($(xReg)) + row * $(nCol) + cols)
    result := tl.max(values, axis=0)
    tl.store(tl.ptr($(yReg)) + row, result)
  }

/-! ## Math denotations -/

/-- Mathematical row-wise sum: just the `Finset.sum` over the tile. -/
noncomputable def rowWiseSumSpec {N : Nat} (xs : Fin N → ℝ) : ℝ :=
  ∑ i, xs i

/-- Mathematical row-wise max via Mathlib's `Finset.sup'`.

The non-empty hypothesis mirrors `Tile.reduceMax`, which is only defined on
`tile (n + 1)` because `Finset.sup'` requires a non-empty index set. -/
noncomputable def rowWiseMaxSpec {N : Nat} (h : 0 < N) (xs : Fin N → ℝ) : ℝ :=
  match N, h, xs with
  | _ + 1, _, xs => Finset.univ.sup' Finset.univ_nonempty xs

/-! ## Kernel correctness -/

end VeriTile.Examples
