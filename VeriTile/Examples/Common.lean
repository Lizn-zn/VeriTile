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

/-- Cast `Fin k` to `Fin n` when `k ≤ n`.

Used by prefix-style proofs that compare a first-`k` recurrence against a
length-`n` input vector. -/
def castFin {n k : Nat} (h : k ≤ n) (i : Fin k) : Fin n :=
  ⟨i.val, lt_of_lt_of_le i.isLt h⟩

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
program — the pattern shared by row-wise reductions and
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

/-! ## Generic ND tile-load / observe

`InputAt` and `observeTileAt` are parameterized by an arbitrary
`offsetFn : TileIndex shape → Nat`, so the same predicates work for any rank
and any memory layout (1D linear, 2D row-major with padding, transposed,
contiguous ND, …). Concrete kernels supply the offset function that matches
the addressing they emit. The 1D-specific `InputLoadedAt` / `observeAt`
above are kept as ergonomic shorthands for the common
`base + i.val` pattern. -/

/-- Region `region` holds an ND tile `xs` at the addresses given by
`offsetFn`. -/
def InputAt {shape : TileShape} (s : BlockState) (region : RegionName)
    (offsetFn : TileIndex shape → Nat)
    (xs : TileIndex shape → ℝ) : Prop :=
  ∀ idx : TileIndex shape, s.mem region (offsetFn idx) = xs idx

/-- Read the cell at `offsetFn idx` of an ND tile from the optional final
state of `exec`. -/
noncomputable def observeTileAt {shape : TileShape}
    (sf : Option BlockState) (region : RegionName)
    (offsetFn : TileIndex shape → Nat) (idx : TileIndex shape) : Option ℝ :=
  sf.map (·.readMem region (offsetFn idx))

/-! ## Standard offset families

The offset functions below cover the layouts produced by typical Triton
kernels. Each comes with an injectivity theorem that feeds
`scatter_readback_nd` / `scatter_readback_prop_masked_nd`. Add more variants
(transposed, blocked, generic strided, …) here as new kernel patterns
appear. -/

namespace Offset

/-- 1D linear offset `base + i.val`, the canonical `pid * N + arange(N)`
addressing pattern. -/
def linear1D {n : Nat} (base : Nat) : TileIndex [n] → Nat
  | (i, _) => base + i.val

theorem linear1D_inj {n : Nat} (base : Nat) :
    Function.Injective (linear1D (n := n) base) := by
  rintro ⟨a, _⟩ ⟨b, _⟩ hab
  obtain rfl : a = b := Fin.ext (Nat.add_left_cancel hab)
  rfl

/-- 2D row-major offset `base + i.val * rowStride + j.val` with explicit
row stride (the "leading dimension" of the host matrix). When
`rowStride = cols` the tile is fully contiguous; with `rowStride > cols`
the layout has padding between rows, matching strided `tl.load(... +
offs_m[:, None] * stride_m + offs_d[None, :])` patterns. -/
def rowMajor2D {rows cols : Nat} (base rowStride : Nat) :
    TileIndex [rows, cols] → Nat
  | (i, j, _) => base + i.val * rowStride + j.val

/-- Injectivity of the 2D row-major offset map. Requires `cols ≤ rowStride`
so distinct rows do not overlap; the proof recovers `(i, j)` from the
offset via integer division/modulo by `rowStride`. -/
theorem rowMajor2D_inj {rows cols : Nat} (base rowStride : Nat)
    (h : cols ≤ rowStride) :
    Function.Injective (rowMajor2D (rows := rows) (cols := cols) base rowStride) := by
  rintro ⟨a₁, a₂, _⟩ ⟨b₁, b₂, _⟩ hab
  have ha₂ : a₂.val < rowStride := lt_of_lt_of_le a₂.isLt h
  have hb₂ : b₂.val < rowStride := lt_of_lt_of_le b₂.isLt h
  have hSpos : 0 < rowStride := lt_of_le_of_lt (Nat.zero_le _) ha₂
  have hsum : a₁.val * rowStride + a₂.val = b₁.val * rowStride + b₂.val := by
    have h := hab
    simp only [rowMajor2D] at h
    omega
  have h_a₂ : a₂.val = b₂.val := by
    have h_mod : (a₁.val * rowStride + a₂.val) % rowStride
              = (b₁.val * rowStride + b₂.val) % rowStride := by rw [hsum]
    rw [Nat.mul_add_mod_self_right, Nat.mul_add_mod_self_right,
        Nat.mod_eq_of_lt ha₂, Nat.mod_eq_of_lt hb₂] at h_mod
    exact h_mod
  have h_a₁ : a₁.val = b₁.val := by
    have h_div : (a₁.val * rowStride + a₂.val) / rowStride
              = (b₁.val * rowStride + b₂.val) / rowStride := by rw [hsum]
    rw [show a₁.val * rowStride = rowStride * a₁.val from Nat.mul_comm _ _,
        show b₁.val * rowStride = rowStride * b₁.val from Nat.mul_comm _ _,
        Nat.mul_add_div hSpos, Nat.mul_add_div hSpos,
        Nat.div_eq_of_lt ha₂, Nat.div_eq_of_lt hb₂] at h_div
    omega
  obtain rfl : a₁ = b₁ := Fin.ext h_a₁
  obtain rfl : a₂ = b₂ := Fin.ext h_a₂
  rfl

end Offset

/-! ## Backwards-compatible 1D injectivity helper -/

/-- Stride injectivity for the canonical 1D scatter offset
`fun idx : TileIndex [n] => base + idx.1.val`.

Used by every 1D kernel proof feeding `scatter_readback_nd` /
`scatter_readback_prop_masked_nd` after the ND switch. Equivalent to
`Offset.linear1D_inj`; kept as a separate name for the existing call
sites. -/
theorem injective_offset_singleton {n : Nat} (base : Nat) :
    Function.Injective (fun idx : TileIndex [n] => base + idx.1.val) :=
  Offset.linear1D_inj (n := n) base

end VeriTile.Examples
