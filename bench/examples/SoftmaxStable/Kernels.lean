import VeriTile.Triton.DSL

/-!
SoftmaxStable: source programs shared by the real and FP proof files.
The definitions retain their original operation order, precision, casts, masks
and memory effects. Correct.lean interprets the mathematical projection;
FPEquiv.lean establishes equivalence under the indicated atomic assumptions.
These sources use libdevice.exp for exp-sub rewrites: the configured fp32
tl.exp probe failed its bias gate (0.1608954387 ULP > 0.05).
-/

namespace VeriTile.Bench.Examples.SoftmaxStable.Kernels
open VeriTile Triton

def naiveSoftmaxKernel (xReg yReg : RegionName) (B : Nat) : ComputeKernel := triton {
  pid := tl.program_id(0)
  offs := pid * $(B) + tl.arange(0, $(B))
  x := tl.load($(xReg) + offs)
  e := libdevice.exp(x)
  s := tl.sum(e, axis=0)
  y := e / s
  tl.store($(yReg) + offs, (y).to(tl.bfloat16))
}

def stableSoftmaxKernel (xReg yReg : RegionName) (B : Nat) : ComputeKernel := triton {
  pid := tl.program_id(0)
  offs := pid * $(B) + tl.arange(0, $(B))
  x := tl.load($(xReg) + offs)
  m := tl.max(x, axis=0)
  e := libdevice.exp(x - m)
  s := tl.sum(e, axis=0)
  y := e / s
  tl.store($(yReg) + offs, (y).to(tl.bfloat16))
}

end VeriTile.Bench.Examples.SoftmaxStable.Kernels
