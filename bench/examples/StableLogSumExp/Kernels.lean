import VeriTile.Triton.DSL

/-!
StableLogSumExp: source programs shared by the real and FP proof files.
The definitions retain their original operation order, precision, casts, masks
and memory effects. Correct.lean interprets the mathematical projection;
FPEquiv.lean records the pending FP target and its missing atomic assumptions.
These sources use libdevice.exp for exp-sub rewrites: the configured fp32
tl.exp probe failed its bias gate (0.1608954387 ULP > 0.05).
The plain log identities needed for FP equivalence remain unadmitted.
-/

namespace VeriTile.Bench.Examples.StableLogSumExp.Kernels
open VeriTile Triton

def directLSEKernel (xReg yReg : RegionName) (blockSize : Nat) : ComputeKernel := triton {
  pid  := tl.program_id(0)
  offs := pid * $(blockSize) + tl.arange(0, $(blockSize))
  x    := tl.load($(xReg) + offs)
  e    := libdevice.exp(x)
  s    := tl.sum(e, axis=0)
  y    := tl.log(s)
  tl.store($(yReg) + pid, (y).to(tl.bfloat16))
}

/-- Numerically-stable log-sum-exp: subtract the row max first,
`y = m + log Σ exp(x − m)`, stored rounded to bf16. -/
def stableLSEKernel (xReg yReg : RegionName) (blockSize : Nat) : ComputeKernel := triton {
  pid  := tl.program_id(0)
  offs := pid * $(blockSize) + tl.arange(0, $(blockSize))
  x    := tl.load($(xReg) + offs)
  m    := tl.max(x, axis=0)
  e    := libdevice.exp(x - m)
  s    := tl.sum(e, axis=0)
  y    := m + tl.log(s)
  tl.store($(yReg) + pid, (y).to(tl.bfloat16))
}

end VeriTile.Bench.Examples.StableLogSumExp.Kernels
