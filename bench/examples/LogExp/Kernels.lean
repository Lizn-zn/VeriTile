import VeriTile.Triton.DSL

/-!
Elementwise fp32 log-exp elimination on a symbolic tile.
The original computes log1p(expm1(a)) for |a| ≤ 0.5 and log(exp(a)) otherwise;
the optimized version copies the input. The original masks inactive branch
arguments before the libdevice calls. Both preserve the fp32 load/store interface.
The configured tl.exp exp-sub probe failed admission, so this source uses
libdevice.exp. The corresponding real and FP proofs are in Correct.lean and
FPEquiv.lean respectively.
-/

namespace VeriTile.Bench.Examples.LogExp
open Triton

/-- Piecewise log-exp computation, applied elementwise to a symbolic tile. -/
def originalKernel (xReg yReg : RegionName) (blockSize : Nat) : ComputeKernel := triton {
  pid := tl.program_id(0)
  offs := pid * $(blockSize) + tl.arange(0, $(blockSize))
  a := tl.load($(xReg) + offs, dtype=tl.float32)
  near_zero := tl.abs(a) <= 0.5
  small_a := tl.where(near_zero, a, 0.0)
  other_a := tl.where(near_zero, 0.0, a)
  small := libdevice.log1p(libdevice.expm1(small_a))
  other := libdevice.log(libdevice.exp(other_a))
  out := tl.where(near_zero, small, other)
  tl.store($(yReg) + offs, (out).to(tl.float32))
}

/-- Replace the piecewise computation with the loaded input. -/
def optimizedKernel (xReg yReg : RegionName) (blockSize : Nat) : ComputeKernel := triton {
  pid := tl.program_id(0)
  offs := pid * $(blockSize) + tl.arange(0, $(blockSize))
  a := tl.load($(xReg) + offs, dtype=tl.float32)
  tl.store($(yReg) + offs, (a).to(tl.float32))
}

end VeriTile.Bench.Examples.LogExp
