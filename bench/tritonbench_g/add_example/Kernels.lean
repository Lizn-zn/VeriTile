import VeriTile.Triton

/-! `add_kernel` from add_example. The original preserves the complete per-program body
from add_example.py. The candidate changes only the operand order of addition;
this is a rewrite example, with no performance claim. The host launch is outside
the proofs. Element count and block size remain symbolic. -/

namespace VeriTile.Bench.TritonBenchG.AddExample
open VeriTile.Triton

/-- Faithful transcription of `add_example.py`'s `add_kernel`. -/
def add_kernel
    (in_ptr0 in_ptr1 out_ptr : RegionName)
    (n_elements BLOCK_SIZE : Nat) :
    ComputeKernel := triton {
  pid = tl.program_id(axis=0)
  block_start = pid * $(BLOCK_SIZE)
  offsets = block_start + tl.arange(0, $(BLOCK_SIZE))
  mask = offsets < $(n_elements)
  x = tl.load(in_ptr0 + offsets, mask=mask)
  y = tl.load(in_ptr1 + offsets, mask=mask)
  output = x + y
  tl.store(out_ptr + offsets, output, mask=mask)
}

/-- Candidate with the same loads, mask and store, adding y to x. -/
def add_commuted_kernel
    (in_ptr0 in_ptr1 out_ptr : RegionName)
    (n_elements BLOCK_SIZE : Nat) :
    ComputeKernel := triton {
  pid = tl.program_id(axis=0)
  block_start = pid * $(BLOCK_SIZE)
  offsets = block_start + tl.arange(0, $(BLOCK_SIZE))
  mask = offsets < $(n_elements)
  x = tl.load(in_ptr0 + offsets, mask=mask)
  y = tl.load(in_ptr1 + offsets, mask=mask)
  output = y + x
  tl.store(out_ptr + offsets, output, mask=mask)
}

end VeriTile.Bench.TritonBenchG.AddExample
