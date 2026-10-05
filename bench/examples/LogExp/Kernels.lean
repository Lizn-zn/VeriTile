import VeriTile.Triton.DSL

/-!
Elementwise fp32 log-exp rewriting on a symbolic tile.
The original always computes log(exp(a)). The candidate returns a for
0.5 < |a| ≤ 80 and retains the original computation otherwise. Both preserve
the fp32 load/store interface. This is the lane-wise source; the measured GPU
kernel also skips both calls for whole blocks selecting the identity.
The real and FP specifications are in Correct.lean and FPEquiv.lean.
-/

namespace VeriTile.Bench.Examples.LogExp
open Triton

/-- Fixed reference: both libdevice calls and their fp32 rounding are retained. -/
def originalKernel (xReg yReg : RegionName) (blockSize : Nat) : ComputeKernel := triton {
  pid := tl.program_id(0)
  offs := pid * $(blockSize) + tl.arange(0, $(blockSize))
  a := tl.load($(xReg) + offs, dtype=tl.float32)
  out := libdevice.log(libdevice.exp(a))
  tl.store($(yReg) + offs, (out).to(tl.float32))
}

/-- Piecewise candidate with zero-masked inactive arguments. -/
def optimizedKernel (xReg yReg : RegionName) (blockSize : Nat) : ComputeKernel := triton {
  pid := tl.program_id(0)
  offs := pid * $(blockSize) + tl.arange(0, $(blockSize))
  a := tl.load($(xReg) + offs, dtype=tl.float32)
  simplify := (0.5 < tl.abs(a)) & (tl.abs(a) <= 80.0)
  fallback_a := tl.where(simplify, 0.0, a)
  fallback := libdevice.log(libdevice.exp(fallback_a))
  out := tl.where(simplify, a, fallback)
  tl.store($(yReg) + offs, (out).to(tl.float32))
}


end VeriTile.Bench.Examples.LogExp
