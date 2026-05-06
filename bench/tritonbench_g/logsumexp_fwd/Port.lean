import VeriTile.Triton.Core
import VeriTile.Triton.DSL

namespace VeriTile.Bench.TritonBenchG.LogSumExpFwd

open VeriTile.Triton

/-- Basic VeriTile DSL port of `logsumexp_fwd.py`'s `logsumexp_fwd_kernel`
without the optional scale branch. -/
def logsumexpFwdKernel (xReg zReg : RegionName) (d blockSize : Nat) :
    ComputeKernel := triton {
  iN := tl.program_id(0)
  iD := tl.program_id(1)
  offsD := iD * $(blockSize) + tl.arange(0, $(blockSize))
  maskD := offsD < $(d)
  x := tl.load($(xReg) + iN * $(d) + offsD, mask=maskD, other=-inf)
  m := tl.max(x, axis=0)
  z := tl.log(tl.sum(tl.exp(x - m), axis=0)) + m
  tl.store($(zReg) + iN * tl.cdiv($(d), $(blockSize)) + iD, z)
}

end VeriTile.Bench.TritonBenchG.LogSumExpFwd
