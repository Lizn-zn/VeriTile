import VeriTile.Triton.Core
import VeriTile.Triton.DSL

namespace VeriTile.Bench.TritonBenchG.SinKernel

open VeriTile.Triton

/-- Basic VeriTile DSL port of `sin_kernel.py`'s `kernel_function`.
The upstream spelling is `tl.math.sin`; the VeriTile surface uses `tl.sin`. -/
def kernelFunction (xReg outReg : RegionName) (nElements blockSize : Nat) :
    ComputeKernel := triton {
  pid := tl.program_id(0)
  offsets := pid * $(blockSize) + tl.arange(0, $(blockSize))
  mask := offsets < $(nElements)
  x := tl.load($(xReg) + offsets, mask=mask)
  out := tl.sin(x)
  tl.store($(outReg) + offsets, out, mask=mask)
}

end VeriTile.Bench.TritonBenchG.SinKernel
