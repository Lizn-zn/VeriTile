import VeriTile.Triton.Core
import VeriTile.Triton.DSL

namespace VeriTile.Bench.TritonBenchG.SinComputation

open VeriTile.Triton

/-- Basic VeriTile DSL port of `sin_computation.py`'s `sin_kernel`. -/
def sinKernel (inReg outReg : RegionName) (nElements blockSize : Nat) :
    ComputeKernel := triton {
  pid := tl.program_id(0)
  offsets := pid * $(blockSize) + tl.arange(0, $(blockSize))
  mask := offsets < $(nElements)
  x := tl.load($(inReg) + offsets, mask=mask)
  out := tl.sin(x)
  tl.store($(outReg) + offsets, out, mask=mask)
}

end VeriTile.Bench.TritonBenchG.SinComputation
