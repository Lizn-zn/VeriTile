import VeriTile.Triton.Core
import VeriTile.Triton.DSL

namespace VeriTile.Bench.TritonBenchG.AddValue

open VeriTile.Triton

/-- Basic VeriTile DSL port of `add_value.py`'s `puzzle1_kernel`. -/
def addValueKernel (xReg outReg : RegionName) (nElements blockSize : Nat) (value : ℝ) :
    ComputeKernel := triton {
  pid := tl.program_id(0)
  offsets := pid * $(blockSize) + tl.arange(0, $(blockSize))
  mask := offsets < $(nElements)
  x := tl.load($(xReg) + offsets, mask=mask)
  out := x + $ℝ(value)
  tl.store($(outReg) + offsets, out, mask=mask)
}

end VeriTile.Bench.TritonBenchG.AddValue
