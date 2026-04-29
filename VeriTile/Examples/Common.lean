/-
VeriTile.Examples.Common

Shared helpers for the parameterized-region kernel correctness pattern:
* `InputLoadedAt`  — region `R` holds the tile `xs` at offsets `[pid*N, pid*N + N)`.
* `observeAt`      — read region `R`'s cell at offset `pid*N + i.val` from
  the optional final state of `exec`.

Both are parameterized by the region name (a `RegionName = String`),
matching the DSL convention that kernels take their buffer regions as
explicit `RegionName` parameters and thread them via `tl.load($(xReg) + …)`
/ `tl.store($(outReg) + …, …)` pointer-like syntax.

These supersede the hardcoded `InputLoaded` ("X") / `observeY` ("Y") used
during early Phase A, which were dropped when we banned the bare-ident
form `tl.load(X + …)` from the DSL.
-/

import VeriTile.Triton.Core
import VeriTile.Triton.Semantics

namespace VeriTile.Examples

open VeriTile.Triton

/-- Region `region` holds tile `xs` at offsets `[pid*N, pid*N + N - 1]`. -/
def InputLoadedAt (s : BlockState) (region : RegionName)
    (N : Nat) (xs : Fin N → ℝ) : Prop :=
  ∀ i : Fin N, s.mem region (s.pid * N + i.val) = xs i

/-- Region `region` holds a feature vector at offsets `[0, N)`.
    Used for per-column parameters such as LayerNorm `γ` and `β`, which are
    shared across rows and are loaded with `tl.arange(0, N)`, not
    `pid * N + tl.arange(0, N)`. -/
def InputFeatureLoadedAt (s : BlockState) (region : RegionName)
    (N : Nat) (xs : Fin N → ℝ) : Prop :=
  ∀ i : Fin N, s.mem region i.val = xs i

/-- Read region `region` at the cell `pid*N + i.val` from the optional
    final `BlockState` of an `exec` call. -/
noncomputable def observeAt
    (sf : Option BlockState) (region : RegionName)
    (N : Nat) (basePid : Nat) (i : Fin N) : Option ℝ :=
  sf.map (·.readMem region (basePid * N + i.val))

/-! ## Row-wise (2D-style) variants

For kernels that gather one *row* of a row-major matrix (row stride
`rowStride`, `blockSize` cells per row) and write a single scalar per
program — the pattern shared by `RowWiseSum`, `RowWiseMax`, and
forthcoming row-wise reductions. Generalizes the predicates above by
separating the row stride (the leading dimension of the matrix in
memory) from the block size (the number of cells gathered). When
`rowStride = blockSize` they collapse to the 1D forms. -/

/-- Region `region` holds the `blockSize`-cell tile `xs` at offsets
    `[s.pid * rowStride, s.pid * rowStride + blockSize)`. -/
def InputRowLoadedAt (s : BlockState) (region : RegionName)
    (rowStride blockSize : Nat) (xs : Fin blockSize → ℝ) : Prop :=
  ∀ i : Fin blockSize, s.mem region (s.pid * rowStride + i.val) = xs i

/-- Read region `region` at the single cell `basePid` from the optional
    final `BlockState` of an `exec` call. Models the
    *single-scalar-per-block* output pattern (`tl.store($(yReg) + row, _)`)
    distinct from the tile-scatter pattern observed by `observeAt`. -/
noncomputable def observeRowAt
    (sf : Option BlockState) (region : RegionName) (basePid : Nat) : Option ℝ :=
  sf.map (·.readMem region basePid)

end VeriTile.Examples
