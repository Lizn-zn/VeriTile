import VeriTile.Triton.DSL

/-!
RowWiseMax: source programs shared by the real and FP proof files.
The definitions retain their original operation order, precision, casts, masks
and memory effects. Correct.lean interprets the mathematical projection;
FPEquiv.lean establishes equivalence under the indicated atomic assumptions.
-/

namespace VeriTile.Bench.Examples.RowWiseMax.Kernels
open VeriTile Triton

/-- Row-wise max over a row-major 2D matrix.

Per `program_id`: gather `blockSize` cells from row `pid` of `xReg` (row
stride `nCol`), reduce by max, and scatter the scalar to `yReg[pid]`. -/
def rowWiseMaxKernel (xReg yReg : RegionName) (nCol blockSize : Nat) : ComputeKernel :=
  triton {
    row    := tl.program_id(0)
    cols   := tl.arange(0, $(blockSize))
    values := tl.load($(xReg) + row * $(nCol) + cols)
    result := tl.max(values, axis=0)
    tl.store($(yReg) + row, result)
  }

/-- Eliminate the two intermediate register bindings without changing the
load, reduction or output address. -/
def inlinedKernel (xReg yReg : RegionName) (nCol blockSize : Nat) : ComputeKernel :=
  triton {
    row    := tl.program_id(0)
    cols   := tl.arange(0, $(blockSize))
    tl.store($(yReg) + row, tl.max(tl.load($(xReg) + row * $(nCol) + cols), axis=0))
  }

end VeriTile.Bench.Examples.RowWiseMax.Kernels
