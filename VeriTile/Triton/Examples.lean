/-
VeriTile.Triton.Examples

Small smoke tests for the typed Triton core.
-/

import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.DSL

namespace VeriTile.Triton.Examples

/-- Smoke test for scalar-pointer load/store syntax. -/
def scalarCopyKernel (xReg yReg : RegionName) : Kernel := triton {
  x := tl.load($(xReg))
  tl.store($(yReg), x)
}

/-- Vector-add kernel with explicit boundary mask. -/
def addKernelMaskedSmoke (xReg yReg outReg : RegionName)
    (blockSize nElements : Nat) : Kernel := triton {
  pid  := tl.program_id(0)
  offs := pid * $(blockSize) + tl.arange(0, $(blockSize))
  mask := offs < $(nElements)
  x    := tl.load($(xReg) + offs, mask=mask, other=0)
  y    := tl.load($(yReg) + offs, mask=mask, other=0)
  out  := x + y
  tl.store($(outReg) + offs, out, mask=mask)
}

/-- Every comparison operator is reachable via the DSL. -/
def comparisonOpsSmoke : Kernel := triton {
  a   := 1
  b   := 2
  r1  := a < b
  r2  := a <= b
  r3  := a == b
  r4  := a > b
  r5  := a >= b
  r6  := a != b
}

/-- `tl.load(p, mask=m)` with no `other=` uses `s.undef` for masked-off lanes. -/
example : evalOp
    (Op.loadMask "X" (Op.constNat 0)
      (Op.lt ComparableDType.nat Broadcast.nil (Op.constNat 0) (Op.constNat 0)))
    { mem := fun _ _ => 100, regs := fun _ _ _ => none
    , pid := 0, undef := fun _ _ => 42 }
    = some (Tile.scalar 42) := by
  rfl

/-- Different `undef` oracles can produce different masked-off load values. -/
example :
    let s1 : BlockState :=
      { mem := fun _ _ => 0, regs := fun _ _ _ => none
      , pid := 0, undef := fun _ _ => 42 }
    let s2 : BlockState :=
      { mem := fun _ _ => 0, regs := fun _ _ _ => none
      , pid := 0, undef := fun _ _ => 99 }
    let maskFalse : Op .bool [] :=
      Op.lt ComparableDType.nat Broadcast.nil (Op.constNat 0) (Op.constNat 0)
    evalOp (Op.loadMask "X" (Op.constNat 0) maskFalse) s1
      ≠ evalOp (Op.loadMask "X" (Op.constNat 0) maskFalse) s2 := by
  change some (Tile.scalar 42) ≠ some (Tile.scalar 99)
  intro h
  injection h with ht
  have hv := congrArg (fun t : Tile .real [] => t.data PUnit.unit) ht
  norm_num at hv

/-! ### `tl.dot` typed-AST smoke tests

These verify that `Op.dot` type-checks at the AST layer (the `K` constraint
forces the inner dim of the LHS to match the outer dim of the RHS). The
DSL macro `tl.dot(a, b)` and its fused-accumulator form
`tl.dot(a, b, acc)` lower into these AST nodes; they will be exercised
end-to-end once 2D load / pointer arithmetic land for FA-1 forward. -/

/-- Rank-2 (FA-1 forward shape): `(M, K) @ (K, N) = (M, N)`. -/
example : (M K N : Nat) → Op .real [M, N] := fun M K N =>
  .dot (batch := []) (Op.ref .real [M, K] "a") (Op.ref .real [K, N] "b")

/-- Fused accumulator form, rank-2: `acc + a @ b`. -/
example : (M K N : Nat) → Op .real [M, N] := fun M K N =>
  .add NumericDType.real (.consSame (.consSame .nil))
    (.dot (batch := []) (Op.ref .real [M, K] "p") (Op.ref .real [K, N] "v"))
    (Op.ref .real [M, N] "acc")

/-- Batched (FA-2 / grouped-GEMM shape): `(B, M, K) @ (B, K, N) = (B, M, N)`. -/
example : (B M K N : Nat) → Op .real [B, M, N] := fun B M K N =>
  .dot (batch := [B])
    (Op.ref .real [B, M, K] "a")
    (Op.ref .real [B, K, N] "b")

/-! ### `tl.expand_dims` surface forms

Two literal slicer postfixes are accepted in this stage:

* `e[:, None]` — `[N] → [N, 1]`, axis 1
* `e[None, :]` — `[N] → [1, N]`, axis 0

Both lower to `Op.expandDim` with the appropriate `Fin (rank + 1)` axis.
Higher-rank inputs raise a macro error. -/

/-- `[:, None]` produces a `[N, 1]` tile. -/
def colExpandSmoke (xReg : RegionName) (N : Nat) : Kernel := triton {
  pid  := tl.program_id(0)
  offs := pid * $(N) + tl.arange(0, $(N))
  x    := tl.load($(xReg) + offs)
  xc   := x[:, None]
}

/-- `[None, :]` produces a `[1, N]` tile. -/
def rowExpandSmoke (xReg : RegionName) (N : Nat) : Kernel := triton {
  pid  := tl.program_id(0)
  offs := pid * $(N) + tl.arange(0, $(N))
  x    := tl.load($(xReg) + offs)
  xr   := x[None, :]
}

/-- `Tile.expandDim` semantics on a literal rank-1 tile: `[3] → [1, 3]`. -/
example :
    let v : Tile .real [3] := Tile.vec (fun i => some (i.val : ℝ))
    let v' := Tile.expandDim ⟨0, by simp⟩ v
    v'.data (⟨0, by decide⟩, ⟨2, by decide⟩, PUnit.unit) =
      v.data (⟨2, by decide⟩, PUnit.unit) := by
  simp [Tile.expandDim, TileShape.dropInsertedIndex]

/-- `Tile.expandDim` axis-1 case: `[3] → [3, 1]`. -/
example :
    let v : Tile .real [3] := Tile.vec (fun i => some (i.val : ℝ))
    let v' := Tile.expandDim ⟨1, by simp⟩ v
    v'.data (⟨1, by decide⟩, ⟨0, by decide⟩, PUnit.unit) =
      v.data (⟨1, by decide⟩, PUnit.unit) := by
  simp [Tile.expandDim, TileShape.dropInsertedIndex]

end VeriTile.Triton.Examples
