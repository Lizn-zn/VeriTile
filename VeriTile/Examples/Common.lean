/-
VeriTile.Examples.Common

Shared example-level helpers for the parameterized-region kernel correctness
pattern. The reusable memory-contract layer (`InputAt`, `Offset`,
`TensorView`) lives in `VeriTile.Triton.Memory`; this file keeps only the
ergonomic shorthands used by worked examples.
-/

import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Memory

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

/-! ## Row-wise (2D-style) variants -/

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

/-! ## Backwards-compatible 1D injectivity helper -/

/-- Stride injectivity for the canonical 1D scatter offset
`fun idx : TileIndex [n] => base + idx.1.val`.

Used by every 1D kernel proof feeding `scatter_readback_nd` /
`scatter_readback_prop_masked_nd` after the ND switch. Equivalent to
`Offset.linear1D_inj`; kept as a separate name for the existing call sites. -/
theorem injective_offset_singleton {n : Nat} (base : Nat) :
    Function.Injective (fun idx : TileIndex [n] => base + idx.1.val) := by
  rintro ⟨a, _⟩ ⟨b, _⟩ hab
  obtain rfl : a = b := Fin.ext (Nat.add_left_cancel hab)
  rfl

end VeriTile.Examples
