import VeriTile.Triton.DSL

/-!
Welford: source programs shared by the real and FP proof files.
The definitions retain their original operation order, precision, casts, masks
and memory effects. Correct.lean interprets the mathematical projection;
FPEquiv.lean establishes equivalence under the indicated atomic assumptions.
-/

namespace VeriTile.Bench.Examples.Welford.Kernels
open VeriTile Triton

/-- Two-pass mean/variance kernel: mean with one `tl.sum`, then population
variance with a second `tl.sum` over squared deviations. -/
def twopassWelfordKernel (xReg meanReg varReg : RegionName)
    (blockSize rowStride : Nat) : ComputeKernel := triton {
  pid    := tl.program_id(0)
  offs   := pid * $(rowStride) + tl.arange($(blockSize))
  x      := tl.load($(xReg) + offs)
  s_x    := tl.sum(x)
  μ      := s_x / tl.toReal($(blockSize))
  d      := x - μ
  s_d2   := tl.sum(d * d)
  v      := s_d2 / tl.toReal($(blockSize))
  tl.store($(meanReg), (μ).to(tl.bfloat16))
  tl.store($(varReg), (v).to(tl.bfloat16))
}

/-- Online Welford mean/variance kernel: the same population mean/variance via
Welford's one-pass recurrence inside a `for` loop (running mean `M`, running
sum-of-squared-deviations `S`). -/
def onlineWelfordKernel (xReg meanReg varReg : RegionName)
    (blockSize rowStride : Nat) : ComputeKernel := triton {
  pid := tl.program_id(0)
  M   := 0
  S   := 0
  tl.for i in $(blockSize) {
    xi     := tl.load($(xReg) + (pid * $(rowStride) + i))
    delta  := xi - M
    M      := M + delta / (tl.toReal(i) + 1)
    delta2 := xi - M
    S      := S + delta * delta2
  }
  tl.store($(meanReg), (M).to(tl.bfloat16))
  tl.store($(varReg), (S / tl.toReal($(blockSize))).to(tl.bfloat16))
}

end VeriTile.Bench.Examples.Welford.Kernels
