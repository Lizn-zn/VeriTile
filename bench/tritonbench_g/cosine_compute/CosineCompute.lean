import VeriTile.Triton.Core
import VeriTile.Triton.DSL

namespace VeriTile.Bench.TritonBenchG.CosineCompute

open VeriTile.Triton

/-- Basic VeriTile DSL port of `cosine_compute.py`'s `cos_func`. -/
def cosKernel (inReg outReg : RegionName) (nElements blockSize : Nat) :
    ComputeKernel := triton {
  offsets := tl.program_id(0) * $(blockSize) + tl.arange(0, $(blockSize))
  mask := offsets < $(nElements)
  x := tl.load($(inReg) + offsets, mask=mask)
  out := tl.cos(x)
  tl.store($(outReg) + offsets, out, mask=mask)
}

end VeriTile.Bench.TritonBenchG.CosineCompute
