import VeriTile.Triton.DSL

/-!
StableLogSumExp: source programs shared by the real and FP proof files.
The original direct and shifted tl.log programs remain explicit. The additional
candidate composes conditional tl.log rewrites. Correct.lean proves all
three mathematical projections; FPEquiv.lean proves the candidate against the
unchanged direct reference. The older unconditional shift is not certified.
These sources use libdevice.exp for exp-sub rewrites: the configured fp32
tl.exp probe failed its bias gate (0.1608954387 ULP > 0.05).
The new log atoms do not equate tl.log with libdevice.log, so they cannot
silently change the original reference's intrinsic.
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

/-- Candidate composed from `log_mul_split .tl` and `log_exp_cancel .tl`.
All logs retain tl.log, exactly as in the original direct reference. The rounded product keeps
its direct log on [0.5, 2]; the split branch eliminates log(exp(m)) only when
0.5 < |m| ≤ 80. Unused log arguments are one and unused exp arguments are zero.
The source preserves both conditional fallbacks and the final bf16 cast. -/
def candidateLSEKernel (xReg yReg : RegionName) (blockSize : Nat) : ComputeKernel := triton {
  pid := tl.program_id(0)
  offs := pid * $(blockSize) + tl.arange(0, $(blockSize))
  x := tl.load($(xReg) + offs)
  m := tl.max(x, axis=0)
  e := libdevice.exp(x - m)
  s := tl.sum(e, axis=0)
  p := s * libdevice.exp(m)
  keep := (p >= 0.5) & (p <= 2.0)
  direct := tl.log(tl.where(keep, p, 1.0))
  split_m := tl.where(keep, 0.0, m)
  simplify := (0.5 < tl.abs(split_m)) & (tl.abs(split_m) <= 80.0)
  fallback := tl.log(libdevice.exp(tl.where(simplify, 0.0, split_m)))
  log_em := tl.where(simplify, split_m, fallback)
  split := tl.log(tl.where(keep, 1.0, s)) + log_em
  y := tl.where(keep, direct, split)
  tl.store($(yReg) + pid, (y).to(tl.bfloat16))
}

end VeriTile.Bench.Examples.StableLogSumExp.Kernels
