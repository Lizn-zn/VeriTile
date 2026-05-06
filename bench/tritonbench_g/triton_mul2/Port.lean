import VeriTile.Triton.Core
import VeriTile.Triton.DSL

namespace VeriTile.Bench.TritonBenchG.TritonMul2

open VeriTile.Triton

/-- Basic VeriTile DSL port of `triton_mul2.py`'s out-of-place `mul2_kernel`. -/
def mul2Kernel (inReg outReg : RegionName) (nElements blockSize : Nat) :
    ComputeKernel := triton {
  pid := tl.program_id(0)
  offsets := pid * $(blockSize) + tl.arange(0, $(blockSize))
  mask := offsets < $(nElements)
  x := tl.load($(inReg) + offsets, mask=mask)
  out := 2 * x
  tl.store($(outReg) + offsets, out, mask=mask)
}

/-- Basic VeriTile DSL port of `triton_mul2.py`'s in-place `mul2_inplace_kernel`. -/
def mul2InplaceKernel (reg : RegionName) (nElements blockSize : Nat) :
    ComputeKernel := triton {
  pid := tl.program_id(0)
  offsets := pid * $(blockSize) + tl.arange(0, $(blockSize))
  mask := offsets < $(nElements)
  x := tl.load($(reg) + offsets, mask=mask)
  out := 2 * x
  tl.store($(reg) + offsets, out, mask=mask)
}

end VeriTile.Bench.TritonBenchG.TritonMul2
