import VeriTile.Triton.Core
import VeriTile.Triton.DSL

namespace VeriTile.Bench.TritonBenchG.SquareMatrix

open VeriTile.Triton

/-- Basic VeriTile DSL port of `square_matrix.py`'s row-wise `square_kernel`. -/
def squareKernel
    (inputReg outReg : RegionName)
    (inputRowStride outputRowStride nCols blockSize : Nat) :
    ComputeKernel := triton {
  row := tl.program_id(0)
  colOffsets := tl.arange(0, $(blockSize))
  mask := colOffsets < $(nCols)
  inputOffsets := row * $(inputRowStride) + colOffsets
  x := tl.load($(inputReg) + inputOffsets, mask=mask, other=-inf)
  y := x * x
  outputOffsets := row * $(outputRowStride) + colOffsets
  tl.store($(outReg) + outputOffsets, y, mask=mask)
}

end VeriTile.Bench.TritonBenchG.SquareMatrix
