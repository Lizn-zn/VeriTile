import VeriTile.Triton.DSL

/-!
OnlineSoftmax: source programs shared by the real and FP proof files.
The definitions retain their original operation order, precision, casts, masks
and memory effects. Correct.lean interprets the mathematical projection;
FPEquiv.lean establishes equivalence under the indicated atomic assumptions.
These sources use libdevice.exp for exp-sub rewrites: the configured fp32
tl.exp probe failed its bias gate (0.1608954387 ULP > 0.05).
-/

namespace VeriTile.Bench.Examples.OnlineSoftmax.Kernels
open VeriTile Triton

/-- Stable softmax: subtract the max before exponentiating.
-/
def stableSoftmaxKernel (xReg yReg : RegionName) (blockSize : Nat) : ComputeKernel := triton {
  pid  := tl.program_id(0)
  offs := pid * $(blockSize) + tl.arange(0, $(blockSize))
  x    := tl.load($(xReg) + offs)
  m    := tl.max(x, axis=0)
  e    := libdevice.exp(x - m)
  s    := tl.sum(e, axis=0)
  y    := e / s
  tl.store($(yReg) + offs, y)
}

def batchSoftmaxKernel (xReg yReg : RegionName) (N : Nat) : ComputeKernel :=
  stableSoftmaxKernel xReg yReg N

/-- First-pass helper; the complete onlineSoftmaxKernel below also writes y. -/
def onlineNormalizerKernel (xReg _yReg : RegionName) (N : Nat) : ComputeKernel := triton {
  pid := tl.program_id(0)
  m   := -inf
  l   := 0
  tl.for i in $(N) {
    xi    := tl.load($(xReg) + (pid * $(N) + i))
    m_new := tl.max(m, xi)
    l     := libdevice.exp(m - m_new) * l + libdevice.exp(xi - m_new)
    m     := m_new
  }
}

/-- Complete two-pass online softmax. The first pass computes m/l; the
second rereads the input row and stores every normalized output. -/
def onlineSoftmaxKernel (xReg yReg : RegionName) (N : Nat) : ComputeKernel :=
  let code : ComputeKernel := triton {
  pid := tl.program_id(0)
  m := -inf
  l := 0
  tl.for i in $(N) {
    xi := tl.load($(xReg) + (pid * $(N) + i))
    m_new := tl.max(m, xi)
    l := libdevice.exp(m - m_new) * l + libdevice.exp(xi - m_new)
    m := m_new
  }
  pid := tl.program_id(0)
  offs := pid * $(N) + tl.arange(0, $(N))
  x := tl.load($(xReg) + offs)
  e := libdevice.exp(x - m)
  y := e / l
  tl.store($(yReg) + offs, y)
}
  -- Both passes read the same input port.
  .mk [xReg] [yReg] code.surfaceBody

end VeriTile.Bench.Examples.OnlineSoftmax.Kernels
