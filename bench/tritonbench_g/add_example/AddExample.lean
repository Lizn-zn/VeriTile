import VeriTile.Triton.Core
import VeriTile.Triton.DSL

namespace VeriTile.Bench.TritonBenchG.AddExample

open VeriTile.Triton

/-- Basic VeriTile DSL port of `add_example.py`'s `add_kernel`. -/
def addKernel (in0Reg in1Reg outReg : RegionName) (nElements blockSize : Nat) :
    ComputeKernel := triton {
  pid := tl.program_id(0)
  blockStart := pid * $(blockSize)
  offsets := blockStart + tl.arange(0, $(blockSize))
  mask := offsets < $(nElements)
  x := tl.load($(in0Reg) + offsets, mask=mask)
  y := tl.load($(in1Reg) + offsets, mask=mask)
  out := x + y
  tl.store($(outReg) + offsets, out, mask=mask)
}

end VeriTile.Bench.TritonBenchG.AddExample
