import VeriTile.Triton.DSL

/-!
RowWiseSum: source programs shared by the real and FP proof files.
The definitions retain their original operation order, precision, casts, masks
and memory effects. Correct.lean interprets the mathematical projection;
FPEquiv.lean establishes equivalence under the indicated atomic assumptions.
-/

namespace VeriTile.Bench.Examples.RowWiseSum.Kernels
open VeriTile Triton

/-- Row-wise sum over a row-major 2D matrix.

Per `program_id`: gather `blockSize` cells from row `pid` of `xReg` (row
stride `nCol`), sum them, and scatter the scalar to `yReg[pid]`. -/
def rowWiseSumKernel (xReg yReg : RegionName) (nCol blockSize : Nat) : ComputeKernel :=
  triton {
    row    := tl.program_id(0)
    cols   := tl.arange(0, $(blockSize))
    values := tl.load($(xReg) + row * $(nCol) + cols, dtype=tl.float32)
    result := tl.sum(values, axis=0)
    tl.store($(yReg) + row, result)
  }

/-- The same reduction with its input lanes visited in reverse order. -/
def reversedKernel (xReg yReg : RegionName) (nCol blockSize : Nat) : ComputeKernel :=
  let x_ptr : Region .fp32 := ⟨xReg.name⟩
  triton {
    row    := tl.program_id(0)
    cols   := $(blockSize - 1) - tl.arange(0, $(blockSize))
    values := tl.load(x_ptr + row * $(nCol) + cols)
    result := tl.sum(values, axis=0)
    tl.store($(yReg) + row, result)
  }

end VeriTile.Bench.Examples.RowWiseSum.Kernels
