/-
bench/tests/PtrElemDType

Regression: a pointer register rebound across two regions of *different* element
dtypes must give each load the dtype in effect at that load's own program point.

`DSL/Inference.collectPtrElems` is a whole-body pre-pass that records
`pointer name → element dtype` with prepend, and `lookupPtrElem` takes the first
match — so on its own it resolves a name to whichever binding comes **last in the
body**, everywhere in the body, including *before* that binding. The expansion
driver therefore threads the map per statement (`ptrElemsAfterStmt`) exactly as it
already threads `Env`, and this file pins the result.

Two things were wrong before the threading, and both are covered here:

* a load could be given the **wrong** element dtype, not merely be rejected —
  `ptrRebind_load_dtypes` fails without the fix, because `x`'s load came back
  `.nat` while `ptr` still pointed into the `.real` region;
* consequently a kernel that *should* elaborate could be rejected, since the
  wrong dtype propagates into the following arithmetic. `matmul_dequantize_int4`
  and its two siblings in TritonBench-G are blocked exactly this way when the map
  is not threaded (see `bench/tritonbench_coverage.md`).
-/

import VeriTile.Triton

namespace VeriTile.Bench.Tests.PtrElemDType

open VeriTile.Triton

/-- `ptr` points into the `.real` region `S`, is loaded, then is rebound to point
into the `.nat` region `Z` and loaded again. This is the shape the three
`matmul_dequant*` kernels use to reload per-group scales and zeros. -/
def ptrRebindKernel (Z : Region .nat) (S : RegionName) (BN : Nat) :
    ComputeKernel := triton {
  sp = S + tl.arange(0, $(BN))
  zp = Z + tl.arange(0, $(BN))
  ptr = sp + $(1)
  x = tl.load(ptr)
  ptr = zp + $(1)
  y = tl.load(ptr)
  tl.store(sp, x)
}

/-- The dtype a statement assigns at, if it is an assignment. -/
private def assignDType? : Stmt → Option TileDType
  | .assign d _ _ _ => some d
  | _ => none

/-- **★ The regression.** The first load is `.real` (its pointer still points into
`S`); the second is `.nat`. Without the per-statement threading of `PtrElems`
both came back `.nat`, because the later `ptr = zp + 1` won the lookup at *both*
load sites. -/
theorem ptrRebind_load_dtypes (Z : Region .nat) (S : RegionName) (BN : Nat) :
    ((ptrRebindKernel Z S BN).toAlgKernel.body[3]?.bind assignDType?
        = some .real)
      ∧ ((ptrRebindKernel Z S BN).toAlgKernel.body[5]?.bind assignDType?
        = some .nat) := by
  constructor <;> rfl

/-- The same rebinding performed inside a `for` body, which is where the
`matmul_dequant*` kernels do it: the per-group reload sits in the K loop. -/
def ptrRebindInLoopKernel (Z : Region .nat) (S : RegionName) (BN n : Nat) :
    ComputeKernel := triton {
  sp = S + tl.arange(0, $(BN))
  zp = Z + tl.arange(0, $(BN))
  for k in range($(0), $(n), $(1)) {
    ptr = sp + k
    x = tl.load(ptr)
    ptr = zp + k
    y = tl.load(ptr)
    tl.store(sp, x)
  }
}

/-- The body of a `forRange`, if that is what the statement is. All three loop
bounds here are static antiquotes, so the DSL lowers this to `Stmt.forRange` and
not `Stmt.forRangeDyn`. -/
private def loopBody? : Stmt → Option (List Stmt)
  | .forRange _ _ _ _ b => some b
  | _ => none

/-- Inside the loop body the two loads keep their own dtypes too. -/
theorem ptrRebindInLoop_load_dtypes (Z : Region .nat) (S : RegionName)
    (BN n : Nat) :
    (((ptrRebindInLoopKernel Z S BN n).toAlgKernel.body[2]?.bind loopBody?).bind
        (fun b => b[1]?.bind assignDType?) = some .real)
      ∧ (((ptrRebindInLoopKernel Z S BN n).toAlgKernel.body[2]?.bind loopBody?).bind
        (fun b => b[3]?.bind assignDType?) = some .nat) := by
  constructor <;> rfl

end VeriTile.Bench.Tests.PtrElemDType
