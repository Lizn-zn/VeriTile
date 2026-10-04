import VeriTile.Triton.DSL

/-!
FloatDTypeSoftmax: source programs shared by the real and FP proof files.
The definitions retain their original operation order, precision, casts, masks
and memory effects. Correct.lean interprets the mathematical projection;
FPEquiv.lean establishes equivalence under the indicated atomic assumptions.
These sources use libdevice.exp for exp-sub rewrites: the configured fp32
tl.exp probe failed its bias gate (0.1608954387 ULP > 0.05).
-/

namespace VeriTile.Bench.Examples.FloatDTypeSoftmax.Kernels
open VeriTile Triton

def floatStableSoftmaxKernel (xReg yReg : RegionName) (blockSize : Nat) : ComputeKernel := triton {
  pid  := tl.program_id(0)
  offs := pid * $(blockSize) + tl.arange(0, $(blockSize))
  x32  := tl.load($(xReg) + offs, dtype=tl.float32)
  x    := x32.to(tl.float64)
  m    := tl.max(x, axis=0)
  e    := libdevice.exp(x - m)
  s    := tl.sum(e, axis=0)
  y    := e / s
  tl.store($(yReg) + offs, (y).to(tl.float32))
}

/-- Optimized stable softmax: precompute `1 / s` and use multiplication,
saving per-lane divisions versus `floatStableSoftmaxKernel`. Both implementations are interpreted over the reals for correctness. -/
def floatSoftmaxRecipKernel (xReg yReg : RegionName) (blockSize : Nat) : ComputeKernel := triton {
  pid    := tl.program_id(0)
  offs   := pid * $(blockSize) + tl.arange(0, $(blockSize))
  x32    := tl.load($(xReg) + offs, dtype=tl.float32)
  x      := x32.to(tl.float64)
  m      := tl.max(x, axis=0)
  e      := libdevice.exp(x - m)
  s      := tl.sum(e, axis=0)
  inv_s  := 1 / s
  y      := e * inv_s
  tl.store($(yReg) + offs, (y).to(tl.float32))
}

/-- Shared execution prefix. Binding the fp32 load before the fp64 cast
preserves both precision boundaries in the FP interpretation. -/
def commonPrefix (xReg : RegionName) (B : Nat) : ComputeKernel := triton {
  pid  := tl.program_id(0)
  offs := pid * $(B) + tl.arange(0, $(B))
  x32  := tl.load($(xReg) + offs, dtype=tl.float32)
  x    := x32.to(tl.float64)
  m    := tl.max(x, axis=0)
  e    := libdevice.exp(x - m)
  s    := tl.sum(e, axis=0)
}

def originalKernel (xReg yReg : RegionName) (B : Nat) : ComputeKernel :=
  floatStableSoftmaxKernel xReg yReg B

def reciprocalKernel (xReg yReg : RegionName) (B : Nat) : ComputeKernel :=
  floatSoftmaxRecipKernel xReg yReg B

end VeriTile.Bench.Examples.FloatDTypeSoftmax.Kernels
