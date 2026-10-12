import VeriTile.Triton

/-! Masked doubling, with ordinary output and in-place output variants. Each
candidate changes only `2 * x` to `x * 2`; the loads, masks and stores remain.
Both sizes are symbolic. FPEquiv.lean supplies the fp32 profile for unannotated
floating operations; Correct.lean interprets them over the reals. These are
per-program contracts, with no host-launch or performance claim. -/

namespace VeriTile.Bench.TritonBenchG.TritonMul2
open VeriTile.Triton

/-- Faithful transcription of `triton_mul2.py`'s `mul2_kernel`.
The Python constexpr tile size is a Lean Nat parameter. -/
def mul2_kernel
    (in_ptr0 out_ptr : RegionName)
    (n_elements BLOCK_SIZE : Nat) :
    ComputeKernel := triton {
  pid = tl.program_id(axis=0)
  block_start = pid * $(BLOCK_SIZE)
  offsets = block_start + tl.arange(0, $(BLOCK_SIZE))
  mask = offsets < $(n_elements)
  x = tl.load(in_ptr0 + offsets, mask=mask)
  output = 2 * x
  tl.store(out_ptr + offsets, output, mask=mask)
}

/-- Faithful transcription of `triton_mul2.py`'s `mul2_inplace_kernel`. -/
def mul2_inplace_kernel
    (ptr : RegionName)
    (n_elements BLOCK_SIZE : Nat) :
    ComputeKernel := triton {
  pid = tl.program_id(axis=0)
  block_start = pid * $(BLOCK_SIZE)
  offsets = block_start + tl.arange(0, $(BLOCK_SIZE))
  mask = offsets < $(n_elements)
  x = tl.load(ptr + offsets, mask=mask)
  output = 2 * x
  tl.store(ptr + offsets, output, mask=mask)
}

/-- Commuted multiplication with a separate output argument. -/
def mul2_commuted_kernel
    (in_ptr0 out_ptr : RegionName)
    (n_elements BLOCK_SIZE : Nat) :
    ComputeKernel := triton {
  pid = tl.program_id(axis=0)
  block_start = pid * $(BLOCK_SIZE)
  offsets = block_start + tl.arange(0, $(BLOCK_SIZE))
  mask = offsets < $(n_elements)
  x = tl.load(in_ptr0 + offsets, mask=mask)
  output = x * 2
  tl.store(out_ptr + offsets, output, mask=mask)
}

/-- Commuted multiplication writing back to the input buffer. -/
def mul2_inplace_commuted_kernel
    (ptr : RegionName)
    (n_elements BLOCK_SIZE : Nat) :
    ComputeKernel := triton {
  pid = tl.program_id(axis=0)
  block_start = pid * $(BLOCK_SIZE)
  offsets = block_start + tl.arange(0, $(BLOCK_SIZE))
  mask = offsets < $(n_elements)
  x = tl.load(ptr + offsets, mask=mask)
  output = x * 2
  tl.store(ptr + offsets, output, mask=mask)
}

end VeriTile.Bench.TritonBenchG.TritonMul2
