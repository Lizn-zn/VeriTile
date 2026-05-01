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

/-- ND strided offset: `base + Σⱼ idxⱼ * stridesⱼ`. The `strides` list aligns
positionally with `shape`; surplus shape dims (when `strides` is shorter) are
treated as having stride `0`, which is malformed but kept structurally total
so the function is well-defined for any input. Concrete kernels supply
matching-length strides. -/
def strided : (shape : TileShape) → (strides : List Nat) → (base : Nat) →
    TileIndex shape → Nat
  | [],          _,        base, _              => base
  | _ :: _,      [],        base, _              => base
  | _ :: rest,  s :: ss,   base, (i, idxRest)  =>
      strided rest ss (base + i.val * s) idxRest

/-- Canonical contiguous (row-major, innermost-stride-1) strides derived from
a shape: `[shape[1] * shape[2] * …, shape[2] * …, …, shape[k], 1]`. -/
def contigStrides : TileShape → List Nat
  | []        => []
  | _ :: rest => (rest.foldr (· * ·) 1) :: contigStrides rest

/-- Fully-contiguous ND row-major offset: every byte of the tile lives in a
single contiguous block starting at `base`. The strides come from
`contigStrides`. -/
def contig (shape : TileShape) (base : Nat) : TileIndex shape → Nat :=
  strided shape (contigStrides shape) base

/-- 1D linear offset `base + i.val` — the canonical `pid * N + arange(N)`
addressing pattern. Defined as `Offset.strided [n] [1] base` so it
participates in the same general ND-strided framework. -/
def linear1D {n : Nat} (base : Nat) : TileIndex [n] → Nat :=
  strided [n] [1] base

/-- 2D row-major offset with an explicit leading-dimension row stride:
`base + i.val * rowStride + j.val`. When `rowStride = cols` the tile is
fully contiguous (matches `Offset.contig [rows, cols]`); with
`rowStride > cols` the layout has padding between rows, matching strided
`tl.load(... + offs_m[:, None] * stride_m + offs_d[None, :])` patterns. -/
def rowMajor2D {rows cols : Nat} (base rowStride : Nat) :
    TileIndex [rows, cols] → Nat :=
  strided [rows, cols] [rowStride, 1] base

/-! ### Injectivity theorems

`linear1D_inj` (1D) and `rowMajor2D_inj` (2D) are standalone proofs for
the patterns Tier 1/2 + FA-1 forward kernels currently use, with their
own bespoke div/mod recovery arguments. The general ND `strided_inj`
(below, parameterized by `StridesValid`) covers the same patterns and
beyond — Step 1+ FA-1 4D layouts feed it directly — but the 1D / 2D
proofs remain independent rather than being rewritten as corollaries:
no semantic dependency, just a documentation overlap. -/

theorem linear1D_inj {n : Nat} (base : Nat) :
    Function.Injective (linear1D (n := n) base) := by
  rintro ⟨a, _⟩ ⟨b, _⟩ hab
  have h : base + a.val * 1 + 0 = base + b.val * 1 + 0 := by
    simpa [linear1D, strided] using hab
  obtain rfl : a = b := Fin.ext (by omega)
  rfl

/-- Injectivity of the 2D row-major offset map. Requires `cols ≤ rowStride`
so distinct rows do not overlap; recovers `(i, j)` from the offset via
integer division/modulo by `rowStride`. -/
theorem rowMajor2D_inj {rows cols : Nat} (base rowStride : Nat)
    (h : cols ≤ rowStride) :
    Function.Injective (rowMajor2D (rows := rows) (cols := cols) base rowStride) := by
  rintro ⟨a₁, a₂, _⟩ ⟨b₁, b₂, _⟩ hab
  have ha₂ : a₂.val < rowStride := lt_of_lt_of_le a₂.isLt h
  have hb₂ : b₂.val < rowStride := lt_of_lt_of_le b₂.isLt h
  have hSpos : 0 < rowStride := lt_of_le_of_lt (Nat.zero_le _) ha₂
  have hsum : a₁.val * rowStride + a₂.val = b₁.val * rowStride + b₂.val := by
    have h := hab
    simp only [rowMajor2D, strided] at h
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

/-! ### General ND strided injectivity

`strided_inj` proves `Function.Injective (Offset.strided shape strides base)`
under a `StridesValid` hypothesis: each dimension's stride strictly
dominates the maximum offset reachable through the inner dimensions.
This is the standard "non-overlapping strides" condition for layout-aware
addressing — sufficient, but not the weakest possible (see `StridesValid`'s
note on the `d = 0` edge case). Step 1+ FA-1 4D layouts and similar ND
kernels feed this directly; `linear1D_inj` / `rowMajor2D_inj` above remain
as independent proofs for their existing call sites. -/

/-- Maximum offset (relative to `base`) reachable by `strided shape strides`,
given by `Σⱼ (dⱼ - 1) * sⱼ`. Defined recursively in lockstep with
`strided`, so the trailing-no-strides case contributes `0` (matching
`strided`'s constant-`base` fallback). -/
def maxOffset : TileShape → List Nat → Nat
  | [], _              => 0
  | _ :: _, []         => 0
  | d :: ds, s :: ss   => (d - 1) * s + maxOffset ds ss

/-- Layout-validity: the strides positionally match the shape and define
a non-overlapping addressing scheme. Each dimension's stride must
strictly dominate the maximum offset reachable through the inner
dimensions (equivalently, the inner block fits within one stride step).

This is a *sufficient* condition for `strided_inj`, not the weakest one.
With `Nat.sub` truncation, `d = 0` collapses `(d - 1) * s` to `0`, so
the predicate happens to require positive strides for outer dimensions
even when the inner domain is empty (and injectivity would hold
vacuously). FA-1 and current Tier 2 kernels never hit this edge case
because all dims are positive; if a future caller needs the weakest
condition, refine the predicate to skip the constraint when any inner
`d = 0`. -/
def StridesValid : TileShape → List Nat → Prop
  | [], _              => True
  | _ :: _, []         => False
  | _ :: ds, s :: ss   => maxOffset ds ss < s ∧ StridesValid ds ss

/-- Lower bound: `strided` always returns at least `base`. -/
theorem base_le_strided : ∀ {shape : TileShape} (strides : List Nat)
    (base : Nat) (idx : TileIndex shape),
    base ≤ strided shape strides base idx
  | [], _, base, _ => le_refl base
  | _ :: _, [], base, _ => le_refl base
  | _ :: ds, s :: ss, base, (i, idxRest) => by
      simp only [strided]
      have := base_le_strided ss (base + i.val * s) idxRest
      omega

/-- Upper bound: `strided` returns at most `base + maxOffset shape strides`.
No layout-validity hypothesis required; follows from `i.val < d`. -/
theorem strided_le_maxOffset : ∀ {shape : TileShape} (strides : List Nat)
    (base : Nat) (idx : TileIndex shape),
    strided shape strides base idx ≤ base + maxOffset shape strides
  | [], _, base, _ => by simp [strided, maxOffset]
  | _ :: _, [], base, _ => by simp [strided, maxOffset]
  | d :: ds, s :: ss, base, (i, idxRest) => by
      simp only [strided, maxOffset]
      have hRec := strided_le_maxOffset ss (base + i.val * s) idxRest
      have hi : i.val ≤ d - 1 := Nat.le_sub_one_of_lt i.isLt
      have hStep : i.val * s ≤ (d - 1) * s := Nat.mul_le_mul_right s hi
      omega

/-- General ND injectivity for `Offset.strided` under `StridesValid`.
The proof is by induction on `shape`, using `base_le_strided` and
`strided_le_maxOffset` to argue that distinct outer indices land in
disjoint windows of width `s`. -/
theorem strided_inj : ∀ {shape : TileShape} {strides : List Nat} (base : Nat),
    StridesValid shape strides →
    Function.Injective (strided shape strides base) := by
  intro shape
  induction shape with
  | nil =>
      intros _strides _base _hValid a b _hEq
      exact Subsingleton.elim a b
  | cons d ds ih =>
      intros strides base hValid
      match strides, hValid with
      | [], hValid => exact absurd hValid id
      | s :: ss, ⟨hMax, hValidRec⟩ =>
          rintro ⟨i, restA⟩ ⟨j, restB⟩ hEq
          simp only [strided] at hEq
          have hLowerA := base_le_strided ss (base + i.val * s) restA
          have hRangeA := strided_le_maxOffset ss (base + i.val * s) restA
          have hLowerB := base_le_strided ss (base + j.val * s) restB
          have hRangeB := strided_le_maxOffset ss (base + j.val * s) restB
          have hi_eq_j : i.val = j.val := by
            rcases lt_trichotomy i.val j.val with hlt | heq | hgt
            · exfalso
              have h1 : base + j.val * s ≤ base + i.val * s + maxOffset ds ss :=
                calc base + j.val * s
                    ≤ strided ds ss (base + j.val * s) restB := hLowerB
                  _ = strided ds ss (base + i.val * s) restA := hEq.symm
                  _ ≤ base + i.val * s + maxOffset ds ss := hRangeA
              have h2 : (i.val + 1) * s ≤ j.val * s := Nat.mul_le_mul_right s hlt
              have h2' : i.val * s + s ≤ j.val * s := by
                rw [Nat.add_mul, Nat.one_mul] at h2; exact h2
              omega
            · exact heq
            · exfalso
              have h1 : base + i.val * s ≤ base + j.val * s + maxOffset ds ss :=
                calc base + i.val * s
                    ≤ strided ds ss (base + i.val * s) restA := hLowerA
                  _ = strided ds ss (base + j.val * s) restB := hEq
                  _ ≤ base + j.val * s + maxOffset ds ss := hRangeB
              have h2 : (j.val + 1) * s ≤ i.val * s := Nat.mul_le_mul_right s hgt
              have h2' : j.val * s + s ≤ i.val * s := by
                rw [Nat.add_mul, Nat.one_mul] at h2; exact h2
              omega
          have hij_fin : i = j := Fin.ext hi_eq_j
          subst hij_fin
          have hRest := ih (base + i.val * s) hValidRec hEq
          exact Prod.ext rfl hRest

end Offset

/-! ## Backwards-compatible 1D injectivity helper -/

/-- Stride injectivity for the canonical 1D scatter offset
`fun idx : TileIndex [n] => base + idx.1.val`.

Used by every 1D kernel proof feeding `scatter_readback_nd` /
`scatter_readback_prop_masked_nd` after the ND switch. Equivalent to
`Offset.linear1D_inj`; kept as a separate name for the existing call
sites. -/
theorem injective_offset_singleton {n : Nat} (base : Nat) :
    Function.Injective (fun idx : TileIndex [n] => base + idx.1.val) := by
  rintro ⟨a, _⟩ ⟨b, _⟩ hab
  obtain rfl : a = b := Fin.ext (Nat.add_left_cancel hab)
  rfl

end VeriTile.Examples
