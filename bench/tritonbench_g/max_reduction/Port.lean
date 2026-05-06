import VeriTile.Triton.Core
import VeriTile.Triton.DSL

namespace VeriTile.Bench.TritonBenchG.MaxReduction

open VeriTile.Triton

/-- Basic VeriTile DSL port of `max_reduction.py`'s first-stage `max_kernel_1`. -/
def maxStage1Kernel (inputReg midReg : RegionName) (m blockSize : Nat) :
    ComputeKernel := triton {
  pid := tl.program_id(0)
  offsets := pid * $(blockSize) + tl.arange(0, $(blockSize))
  mask := offsets < $(m)
  x := tl.load($(inputReg) + offsets, mask=mask, other=-inf)
  maxVal := tl.max(x, axis=0)
  tl.store($(midReg) + pid, maxVal)
}

/-- Basic VeriTile DSL port of `max_reduction.py`'s second-stage `max_kernel_2`. -/
def maxStage2Kernel (midReg outReg : RegionName) (midSize blockMid : Nat) :
    ComputeKernel := triton {
  offsets := tl.arange(0, $(blockMid))
  mask := offsets < $(midSize)
  x := tl.load($(midReg) + offsets, mask=mask, other=-inf)
  maxVal := tl.max(x, axis=0)
  tl.store($(outReg), maxVal)
}

end VeriTile.Bench.TritonBenchG.MaxReduction
