import VeriTile.Triton.Core
import VeriTile.Triton.DSL

namespace VeriTile.Bench.TritonBenchG.VectorAddition

open VeriTile.Triton

/-- Basic VeriTile DSL port of `vector_addition.py`'s `add_kernel`. -/
def addKernel (xReg yReg outReg : RegionName) (nElements blockSize : Nat) :
    ComputeKernel := triton {
  pid := tl.program_id(0)
  blockStart := pid * $(blockSize)
  offsets := blockStart + tl.arange(0, $(blockSize))
  mask := offsets < $(nElements)
  x := tl.load($(xReg) + offsets, mask=mask)
  y := tl.load($(yReg) + offsets, mask=mask)
  out := x + y
  tl.store($(outReg) + offsets, out, mask=mask)
}

end VeriTile.Bench.TritonBenchG.VectorAddition
