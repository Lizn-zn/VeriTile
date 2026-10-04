import VeriTile.Triton.DSL

/-!
SoftmaxReciprocal: source programs shared by the real and FP proof files.
The definitions retain their original operation order, precision, casts, masks
and memory effects. Correct.lean interprets the mathematical projection;
FPEquiv.lean establishes equivalence under the indicated atomic assumptions.
These sources use libdevice.exp for exp-sub rewrites: the configured fp32
tl.exp probe failed its bias gate (0.1608954387 ULP > 0.05).
-/

namespace VeriTile.Bench.Examples.SoftmaxReciprocal.Kernels
open VeriTile Triton

def stableSoftmaxKernel (xReg yReg : RegionName) (blockSize : Nat) : ComputeKernel := triton {
  pid  := tl.program_id(0)
  offs := pid * $(blockSize) + tl.arange(0, $(blockSize))
  x    := tl.load($(xReg) + offs, dtype=tl.float32)
  m    := tl.max(x, axis=0)
  e    := libdevice.exp(x - m)
  s    := tl.sum(e, axis=0)
  y    := e / s
  tl.store($(yReg) + offs, (y).to(tl.bfloat16))
}

/-- Optimized stable softmax: precompute `1 / S` once, then multiply per lane
(`y = e · S⁻¹`), stored bf16. -/
def softmaxRecipKernel (xReg yReg : RegionName) (blockSize : Nat) : ComputeKernel := triton {
  pid    := tl.program_id(0)
  offs   := pid * $(blockSize) + tl.arange(0, $(blockSize))
  x      := tl.load($(xReg) + offs, dtype=tl.float32)
  m      := tl.max(x, axis=0)
  e      := libdevice.exp(x - m)
  s      := tl.sum(e, axis=0)
  inv_s  := 1 / s
  y      := e * inv_s
  tl.store($(yReg) + offs, (y).to(tl.bfloat16))
}

def commonPrefix (xReg : RegionName) (B : Nat) : ComputeKernel := triton {
  pid  := tl.program_id(0)
  offs := pid * $(B) + tl.arange(0, $(B))
  x    := tl.load($(xReg) + offs, dtype=tl.float32)
  m    := tl.max(x, axis=0)
  e    := libdevice.exp(x - m)
  s    := tl.sum(e, axis=0)
}

def originalKernel (xReg yReg : RegionName) (B : Nat) : ComputeKernel :=
  stableSoftmaxKernel xReg yReg B

def reciprocalKernel (xReg yReg : RegionName) (B : Nat) : ComputeKernel :=
  softmaxRecipKernel xReg yReg B

end VeriTile.Bench.Examples.SoftmaxReciprocal.Kernels
