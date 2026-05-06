import VeriTile.Triton.Core
import VeriTile.Triton.DSL

namespace VeriTile.Bench.TritonBenchG.ReluTritonKernel

open VeriTile.Triton

/-- Basic VeriTile DSL port of `relu_triton_kernel.py`'s `relu_kernel`.
Keeps the upstream scalar `if pid == 0` guard. -/
def reluKernel (xReg outReg : RegionName) (nElements blockSize : Nat) :
    ComputeKernel := triton {
  pid := tl.program_id(0)
  offsets := pid * $(blockSize) + tl.arange(0, $(blockSize))
  mask := offsets < $(nElements)
  x := tl.load($(xReg) + offsets, mask=mask)
  result := tl.where(x >= $ℝ((0 : ℝ)), x, $ℝ((0 : ℝ)))
  tl.if pid == $(0) {
    tl.store($(outReg) + offsets, result, mask=mask)
  }
}

end VeriTile.Bench.TritonBenchG.ReluTritonKernel
