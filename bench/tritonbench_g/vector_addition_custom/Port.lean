import VeriTile.Triton.Core
import VeriTile.Triton.DSL

namespace VeriTile.Bench.TritonBenchG.VectorAdditionCustom

open VeriTile.Triton

/-- Basic VeriTile DSL port of `vector_addition_custom.py`'s `_add_kernel`. -/
def addKernel (aReg bReg cReg : RegionName) (size blockSize : Nat) :
    ComputeKernel := triton {
  pid := tl.program_id(0)
  offsets := pid * $(blockSize) + tl.arange(0, $(blockSize))
  mask := offsets < $(size)
  a := tl.load($(aReg) + offsets, mask=mask)
  b := tl.load($(bReg) + offsets, mask=mask)
  out := a + b
  tl.store($(cReg) + offsets, out, mask=mask)
}

end VeriTile.Bench.TritonBenchG.VectorAdditionCustom
