import VeriTile.Triton.DSL

/-!
FlatVectorAdd: source programs shared by the real and FP proof files.
The definitions retain their original operation order, precision, casts, masks
and memory effects. Correct.lean interprets the mathematical projection;
FPEquiv.lean establishes equivalence under the indicated atomic assumptions.
-/

namespace VeriTile.Bench.Examples.FlatVectorAdd.Kernels
open VeriTile Triton

/-- Masked elementwise add. Lanes where `pid * blockSize + i < nElements` are
    loaded, summed, and stored. Lanes outside the bound get Triton's
    `other=None` undefined load value, but the store mask skips those lanes,
    so the undefined values are not observed. -/
def addKernelMasked (xReg yReg outReg : RegionName)
    (blockSize nElements : Nat) : ComputeKernel := triton {
  pid     := tl.program_id(0)
  offsets := pid * $(blockSize) + tl.arange(0, $(blockSize))
  mask    := offsets < $(nElements)
  x       := tl.load($(xReg) + offsets, mask=mask)
  y       := tl.load($(yReg) + offsets, mask=mask)
  output  := x + y
  tl.store($(outReg) + offsets, output, mask=mask)
}

/-- Original FlatVectorAdd addition, transcribed with fp32 regions. -/
def originalKernel (nElements blockSize : Nat) : ComputeKernel :=
  let x_ptr : Region .fp32 := ⟨"x"⟩
  let y_ptr : Region .fp32 := ⟨"y"⟩
  let out_ptr : Region .fp32 := ⟨"out"⟩
  triton {
  pid = tl.program_id(axis=0)
  offsets = pid * $(blockSize) + tl.arange(0, $(blockSize))
  mask = offsets < $(nElements)
  x = tl.load(x_ptr + offsets, mask=mask)
  y = tl.load(y_ptr + offsets, mask=mask)
  output = x + y
  tl.store(out_ptr + offsets, output, mask=mask)
}

/-- The sole rewrite is output = y + x. Addresses/masks are unchanged. -/
def optimizedKernel (nElements blockSize : Nat) : ComputeKernel :=
  let x_ptr : Region .fp32 := ⟨"x"⟩
  let y_ptr : Region .fp32 := ⟨"y"⟩
  let out_ptr : Region .fp32 := ⟨"out"⟩
  triton {
  pid = tl.program_id(axis=0)
  offsets = pid * $(blockSize) + tl.arange(0, $(blockSize))
  mask = offsets < $(nElements)
  x = tl.load(x_ptr + offsets, mask=mask)
  y = tl.load(y_ptr + offsets, mask=mask)
  output = y + x
  tl.store(out_ptr + offsets, output, mask=mask)
}

end VeriTile.Bench.Examples.FlatVectorAdd.Kernels
