import VeriTile.Triton.Core
import VeriTile.Triton.DSL

namespace VeriTile.Bench.TritonBenchG.SoftmaxTriton2

open VeriTile.Triton

/-- Basic VeriTile DSL port of `softmax_triton2.py`'s row-wise `softmax_kernel`. -/
def softmaxKernel
    (inputReg outReg : RegionName)
    (inputRowStride outputRowStride nCols blockSize : Nat) :
    ComputeKernel := triton {
  rowIdx := tl.program_id(0)
  colOffsets := tl.arange(0, $(blockSize))
  mask := colOffsets < $(nCols)
  inputOffsets := rowIdx * $(inputRowStride) + colOffsets
  row := tl.load($(inputReg) + inputOffsets, mask=mask, other=-inf)
  rowMinusMax := row - tl.max(row, axis=0)
  numerator := tl.exp(rowMinusMax)
  denominator := tl.sum(numerator, axis=0)
  softmaxOutput := numerator / denominator
  outputOffsets := rowIdx * $(outputRowStride) + colOffsets
  tl.store($(outReg) + outputOffsets, softmaxOutput, mask=mask)
}

end VeriTile.Bench.TritonBenchG.SoftmaxTriton2
