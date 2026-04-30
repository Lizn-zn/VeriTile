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

example : (M K N : Nat) → Op .real [M, N] := fun M K N =>
  .dot (Op.ref .real [M, K] "a") (Op.ref .real [K, N] "b")

example : (M K N : Nat) → Op .real [M, N] := fun M K N =>
  .add NumericDType.real (.consSame (.consSame .nil))
    (.dot (Op.ref .real [M, K] "p") (Op.ref .real [K, N] "v"))
    (Op.ref .real [M, N] "acc")

end VeriTile.Triton.Examples
