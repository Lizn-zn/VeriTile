import VeriTile.Triton.Core
import VeriTile.Triton.DSL

namespace VeriTile.Bench.TritonBenchG.MatrixTranspose

open VeriTile.Triton

/-- Basic VeriTile DSL port of `matrix_transpose.py`'s `kernel`. -/
def transposeKernel
    (matrixReg outReg : RegionName)
    (matrixStrideX matrixStrideY outStrideX outStrideY sizeM dHead : Nat) :
    ComputeKernel := triton {
  m := tl.arange(0, $(sizeM))
  d := tl.arange(0, $(dHead))
  matrixOffsets := (d[None, :]) * $(matrixStrideY) + (m[:, None]) * $(matrixStrideX)
  outOffsets := (d[None, :]) * $(outStrideX) + (m[:, None]) * $(outStrideY)
  x := tl.load($(matrixReg) + matrixOffsets)
  tl.store($(outReg) + outOffsets, x)
}

end VeriTile.Bench.TritonBenchG.MatrixTranspose
