import VeriTile.Triton

/-! Masked vector-plus-scalar from add_value.py. The candidate swaps the
operands of the addition and retains the same load, mask and store. There is no
performance claim. Both element count and tile size are symbolic. The source's
unannotated arithmetic uses the fp32 profile in FPEquiv.lean; Correct.lean uses
real arithmetic. The host launch is outside these per-program contracts. -/

namespace VeriTile.Bench.TritonBenchG.AddValue
open VeriTile.Triton

/-- Faithful 1:1 transcription of `add_value.py`'s `puzzle1_kernel`.

Allowed mechanical Lean-syntax-only changes:
- Python `BLOCK_SIZE: tl.constexpr` → Lean `Nat` parameter.
- `value` (Lean `ℝ` parameter) injected via `$(...)`. -/
def puzzle1_kernel
    (x_ptr output_ptr : RegionName)
    (N BLOCK_SIZE : Nat) (value : ℝ) :
    ComputeKernel := triton {
  pid = tl.program_id(axis=0)
  block_start = pid * $(BLOCK_SIZE)
  offsets = block_start + tl.arange(0, $(BLOCK_SIZE))
  mask = offsets < $(N)
  x = tl.load(x_ptr + offsets, mask=mask)
  output = x + $(value)
  tl.store(output_ptr + offsets, output, mask=mask)
}

/-- Commuted scalar addition with the original memory operations.
The Lean type annotation selects the scalar data channel; it inserts no cast. -/
def puzzle1_commuted_kernel
    (x_ptr output_ptr : RegionName)
    (N BLOCK_SIZE : Nat) (value : ℝ) :
    ComputeKernel := triton {
  pid = tl.program_id(axis=0)
  block_start = pid * $(BLOCK_SIZE)
  offsets = block_start + tl.arange(0, $(BLOCK_SIZE))
  mask = offsets < $(N)
  x = tl.load(x_ptr + offsets, mask=mask)
  output = $((value : Real)) + x
  tl.store(output_ptr + offsets, output, mask=mask)
}

end VeriTile.Bench.TritonBenchG.AddValue
