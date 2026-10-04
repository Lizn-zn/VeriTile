import VeriTile.Triton.DSL

/-!
FloatDTypeAdd: source programs shared by the real and FP proof files.
The definitions retain their original operation order, precision, casts, masks
and memory effects. Correct.lean interprets the mathematical projection;
FPEquiv.lean establishes equivalence under the indicated atomic assumptions.
-/

namespace VeriTile.Bench.Examples.FloatDTypeAdd.Kernels
open VeriTile Triton

/-- The original kernel retains its fp32 load annotations and output cast. -/
def floatAddKernel (xReg yReg outReg : RegionName) (blockSize : Nat) : ComputeKernel := triton {
  pid := tl.program_id(0)
  offs := pid * $(blockSize) + tl.arange(0, $(blockSize))
  x := tl.load($(xReg) + offs, dtype=tl.float32)
  y := tl.load($(yReg) + offs, dtype=tl.float32)
  out := x + y
  tl.store($(outReg) + offs, (out).round_to(tl.float32))
}

/-- Original FloatDTypeAdd addition, transcribed with fp32 regions. -/
def originalKernel (blockSize : Nat) : ComputeKernel :=
  let x_ptr : Region .fp32 := ⟨"x"⟩
  let y_ptr : Region .fp32 := ⟨"y"⟩
  let out_ptr : Region .fp32 := ⟨"out"⟩
  triton {
  pid = tl.program_id(axis=0)
  offs = pid * $(blockSize) + tl.arange(0, $(blockSize))
  x = tl.load(x_ptr + offs)
  y = tl.load(y_ptr + offs)
  out = x + y
  tl.store(out_ptr + offs, (out).round_to(tl.float32))
}

/-- The sole rewrite is out = y + x. Addresses/masks are unchanged. -/
def optimizedKernel (blockSize : Nat) : ComputeKernel :=
  let x_ptr : Region .fp32 := ⟨"x"⟩
  let y_ptr : Region .fp32 := ⟨"y"⟩
  let out_ptr : Region .fp32 := ⟨"out"⟩
  triton {
  pid = tl.program_id(axis=0)
  offs = pid * $(blockSize) + tl.arange(0, $(blockSize))
  x = tl.load(x_ptr + offs)
  y = tl.load(y_ptr + offs)
  out = y + x
  tl.store(out_ptr + offs, (out).round_to(tl.float32))
}

end VeriTile.Bench.Examples.FloatDTypeAdd.Kernels
